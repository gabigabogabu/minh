#!/usr/bin/env bun
import { createOpenRouter } from "@openrouter/ai-sdk-provider";
import { OpenRouter } from "@openrouter/sdk";
import { stepCountIs, streamText, tool, type ModelMessage } from "ai";
import { appendFile, writeFile } from "node:fs/promises";
import { parseArgs } from "node:util";
import readline from "node:readline";
import * as z from 'zod';
import path from "node:path";

const defaultModel = "openrouter/free";
class UsageError extends Error { }

const shellOutputLimit = 10_000;
const tailText = (text: string) =>
  text.length > shellOutputLimit ? `truncated, last ${shellOutputLimit} chars:\n${text.slice(-shellOutputLimit)}` : text;

const helpText = `Usage:
  minh [-c <chat-file>] [-m <model>] <prompt>
  minh --list-models
Options:
  -c, --chat <path>  Path to the .chat file
  -m, --model <id>   OpenRouter model id (default: openrouter/free)
      --list-models  List OpenRouter model ids
  -h, --help         Show help`;

export function parseCliArgs(args = Bun.argv.slice(2)) {
  const parsed = parseArgs({
    args,
    options: {
      chat: { type: "string", short: "c" },
      help: { type: "boolean", short: "h" },
      model: { type: "string", short: "m" },
      "list-models": { type: "boolean" },
    },
    allowPositionals: true,
  });
  const { values, positionals } = parsed;
  const { chat, help, model, "list-models": listModels } = values;
  const [prompt, extra] = positionals;

  if (help) return { help: true } as const;
  if (listModels)
    return { listModels: true } as const;
  if (prompt === undefined)
    throw new UsageError("Missing required positional argument: prompt");
  if (extra !== undefined)
    throw new UsageError(`Expected one prompt, received ${positionals.length}`);

  return { chatPath: chat, model, prompt };
}

async function createTimestampChat() {
  const next = `./${Date.now()}.chat`;
  await writeFile(next, "", { flag: "wx" });
  return { path: next, defaulted: true };
}

async function resolveChatPath(chatPath?: string) {
  if (chatPath) return { path: chatPath, defaulted: false };
  return createTimestampChat();
}

async function listModels() {
  const openrouter = new OpenRouter({ apiKey: Bun.env.OPENROUTER_API_KEY });
  const models = await openrouter.models.list({ outputModalities: "text" });
  models.data.forEach(m => console.log(`${m.id}\t-\t$/mtok ${m.pricing.prompt} in ${m.pricing.completion} out`));
}

async function loadMessages(chatPath: string, prompt: string) {
  const messages = (await Bun.file(chatPath).text())
    .split("\n")
    .filter(line => line.trim())
    .reduce((msgs, l) => { msgs.push(JSON.parse(l)); return msgs }, [] as ModelMessage[])
  messages.push({ role: "user", content: prompt });
  return messages;
}

const appendMessages = (chatPath: string, messages: ModelMessage[]) =>
  appendFile(chatPath, messages.map(message => JSON.stringify(message)).join("\n") + "\n");

const activeShellGroups = new Set<number>();
const signalExitCodes = { SIGHUP: 129, SIGINT: 130, SIGTERM: 143 } as const;

function killShellGroup(pid: number, signal: NodeJS.Signals = "SIGTERM") {
  try { process.kill(process.platform === "win32" ? pid : -pid, signal); } catch { }
}

for (const signal of Object.keys(signalExitCodes) as Array<keyof typeof signalExitCodes>)
  process.once(signal, () => {
    for (const pid of activeShellGroups) killShellGroup(pid, signal);
    process.exit(signalExitCodes[signal]);
  });

process.once("exit", () => {
  for (const pid of activeShellGroups) killShellGroup(pid);
});

async function askToRun(command: string) {
  await new Promise<void>((resolve) => setImmediate(resolve)) // wait for stdout to flush
  const prompt = `Can I run \`${command}\` (y/N)? `;
  if (!process.stdin.isTTY) {
    process.stderr.write(prompt);
    const rl = readline.createInterface({ input: process.stdin, terminal: false });
    return await new Promise<string>(resolve => {
      let answered = false;
      rl.once("line", line => { answered = true; resolve(line); rl.close(); });
      rl.once("close", () => { if (!answered) resolve(""); });
    });
  }
  const rl = readline.promises.createInterface({ input: process.stdin, output: process.stderr });
  const answer = await rl.question(prompt);
  rl.close()
  return answer;
}

const supervisedShell = `
parent=$1
command=$2
(while kill -0 "$parent" 2>/dev/null; do sleep 1; done; kill -TERM -$$ 2>/dev/null) &
watcher=$!
cleanup() { kill "$watcher" 2>/dev/null; wait "$watcher" 2>/dev/null; }
abort() { trap - HUP INT TERM; cleanup; kill -TERM -$$ 2>/dev/null; exit 143; }
trap abort HUP INT TERM
sh -lc "$command" &
child=$!
wait "$child"
status=$?
cleanup
exit "$status"
`;

const useShell = tool({
  description: `Run a shell command in the current working directory and return stdout, stderr, and exit code. You can run minh again to spawn subagents: \`${Bun.argv[1]} --help\` \`yes | ${Bun.argv[1]} -m <openrouter-model-id> "<prompt>" | tail -100\``,
  inputSchema: z.object({ command: z.string().describe("Shell command to run.") }),
  execute: async ({ command }) => {
    const answer = await askToRun(command);
    if (!/^y(es)?$/i.test(answer.trim())) throw new Error('Rejected by User');
    const proc = Bun.spawn(["sh", "-lc", supervisedShell, "minh-useShell", String(process.pid), command], {
      detached: true,
      stdin: "ignore",
      stdout: "pipe",
      stderr: "pipe",
    });
    activeShellGroups.add(proc.pid);
    const [stdout, stderr, exitCode] = await Promise.all([
      new Response(proc.stdout).text(),
      new Response(proc.stderr).text(),
      proc.exited,
    ]).finally(() => activeShellGroups.delete(proc.pid));

    return { exitCode, stdout: tailText(stdout), stderr: tailText(stderr) };
  },
});

async function runPrompt(args: { chatPath?: string; model?: string; prompt: string }) {
  const { path: chatPath, defaulted: defaultedChat } = await resolveChatPath(args.chatPath);
  const model = args.model ?? defaultModel;

  if (defaultedChat) console.error(`Using chat file: ${chatPath}`);
  if (args.model === undefined) console.error(`Using model: ${model}`);
  console.error(Bun.argv[1])

  const openrouter = createOpenRouter({ apiKey: Bun.env.OPENROUTER_API_KEY });
  const messages = await loadMessages(chatPath, args.prompt);
  const result = streamText({
    model: openrouter(model),
    system: `You are ${model} and running in minh (MINimal Harness) ${Bun.argv[1]}. This chat lives at ${path.join(process.cwd(), chatPath)}. Read files before editing, they might have changed.`,
    messages,
    tools: {useShell},
    stopWhen: stepCountIs(100),
    onError: console.error,
  });

  for await (const part of result.fullStream) {
    if (part.type === "text-delta") process.stdout.write(part.text);

    if (part.type === "reasoning-start") process.stderr.write("\n<thinking>");
    if (part.type === "reasoning-delta") process.stderr.write(part.text);
    if (part.type === "reasoning-end") process.stderr.write("</thinking>\n");

    if (part.type === "tool-call" && part.toolName === 'useShell')
      process.stderr.write(`\n<tool> $ ${(part.input as { command: string }).command} </tool>`);
    if (part.type === "tool-result") {
      const { stdout, stderr, exitCode } = (part.output as { stdout: string; stderr: string; exitCode: number })
      if (stderr) process.stderr.write(`\n<tool-error>Exit ${exitCode}: ${stderr}</tool-error>`)
      if (stdout) process.stderr.write(`\n<tool-result>${stdout}</tool-result>\n`);
    }

    if (part.type === "tool-error")
      process.stderr.write(`\<tool-error> ${part.toolName} failed: ${String(part.error)}</tool-error>`);
    if (part.type === "error")
      throw part.error instanceof Error ? part.error : new Error(String(part.error));
    if (part.type === "abort")
      throw new Error(part.reason ?? "Stream aborted");
  }
  process.stdout.write("\n");
  await appendMessages(chatPath, [
    { role: "user", content: args.prompt },
    ...(await result.response).messages,
  ]);
}

if (import.meta.main) {
  try {
    const args = parseCliArgs();
    if ("help" in args) console.log(helpText);
    else if ("listModels" in args) await listModels();
    else await runPrompt(args);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.error(error instanceof UsageError ? `${message}\n\n${helpText}` : message);
    process.exit(1);
  }
}
