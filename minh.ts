#!/usr/bin/env bun
import { createOpenRouter } from "@openrouter/ai-sdk-provider";
import { OpenRouter } from "@openrouter/sdk";
import { stepCountIs, streamText, tool, type ModelMessage } from "ai";
import { appendFile, readdir, writeFile } from "node:fs/promises";
import { parseArgs } from "node:util";
import readline from "node:readline";
import * as z from 'zod';
import path from "node:path";

const defaultModel = "openrouter/free";
class UsageError extends Error {}

const shellOutputLimit = 10_000;
const tailText = (text: string) =>
  text.length > shellOutputLimit ? `truncated, last ${shellOutputLimit} chars:\n${text.slice(-shellOutputLimit)}` : text;

const helpText = `Usage:
  minh [--new-chat] [-c <chat-file>] [-m <model>] <prompt>
  minh --list-models [filter]
Options:
  -c, --chat <path>  Path to the .chat file
  -m, --model <id>   OpenRouter model id (default: openrouter/free)
      --list-models  List OpenRouter model ids, optionally filtered
      --new-chat     Start a new timestamped .chat file instead of using the latest
  -h, --help         Show help`;

export function parseCliArgs(args = Bun.argv.slice(2)) {
  const parsed = parseArgs({
    args,
    options: {
      chat: { type: "string", short: "c" },
      help: { type: "boolean", short: "h" },
      model: { type: "string", short: "m" },
      "list-models": { type: "boolean" },
      "new-chat": { type: "boolean" },
    },
    allowPositionals: true,
  });
  const { values, positionals } = parsed;
  const { chat, help, model, "list-models": listModels, "new-chat": newChat } = values;
  const [prompt, extra] = positionals;

  if (help) return { help: true } as const;
  if (listModels)
    return { listModels: true, filter: positionals } as const;
  if (prompt === undefined)
    throw new UsageError("Missing required positional argument: prompt");
  if (extra !== undefined)
    throw new UsageError(`Expected one prompt, received ${positionals.length}`);
  if (chat !== undefined && newChat)
    throw new UsageError("--chat and --new-chat cannot be used together");

  return { chatPath: chat, model, prompt, newChat };
}

const timestampChat = /^(\d+)\.chat$/;

async function createTimestampChat() {
  const next = `./${Date.now()}.chat`;
  await writeFile(next, "", { flag: "wx" });
  return { path: next, defaulted: true };
}

async function resolveChatPath(chatPath?: string, newChat = false) {
  if (chatPath) return { path: chatPath, defaulted: false };
  if (newChat) return createTimestampChat();

  const latest = (await readdir(".")).filter(file => timestampChat.test(file)).sort().at(-1);
  if (latest) return { path: `./${latest}`, defaulted: true };

  return createTimestampChat();
}

async function listModels(filter?: string[]) {
  const openrouter = new OpenRouter({ apiKey: Bun.env.OPENROUTER_API_KEY });
  const models = await openrouter.models.list({ outputModalities: "text" });
  const query = filter?.map(f => f.toLowerCase());
  const modelsF = query ? models.data.filter(m => query.some(q => m.id.includes(q))) : models.data
  modelsF.forEach(m => console.log(`${m.id}\t-\t$/mtok ${m.pricing.prompt} in ${m.pricing.completion} out`));
}

async function loadMessages(chatPath: string, prompt: string) {
  const messages = (await Bun.file(chatPath).text())
    .split("\n")
    .filter(line => line.trim())
    .reduce((msgs, l) => {msgs.push(JSON.parse(l)); return msgs}, [] as ModelMessage[])
  messages.push({ role: "user", content: prompt });
  return messages;
}

const appendMessages = (chatPath: string, messages: ModelMessage[]) => 
  appendFile(chatPath, messages.map(message => JSON.stringify(message)).join("\n") + "\n");

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

const makeTools = () => ({
  useShell: tool({
    description: "Run a shell command in the current working directory and return stdout, stderr, and exit code.",
    inputSchema: z.object({
      command: z.string().describe("Shell command to run."),
    }),
    execute: async ({ command }) => {
      const answer = await askToRun(command);
      if (!/^y(es)?$/i.test(answer.trim())) throw new Error('Rejected by User');
      const proc = Bun.spawn(["sh", "-lc", command], {
        stdout: "pipe",
        stderr: "pipe",
      });
      const [stdout, stderr, exitCode] = await Promise.all([
        new Response(proc.stdout).text(),
        new Response(proc.stderr).text(),
        proc.exited,
      ]);

      return { exitCode, stdout: tailText(stdout), stderr: tailText(stderr) };
    },
  }),
});

async function runPrompt(args: { chatPath?: string; model?: string; prompt: string; newChat?: boolean }) {
  const { path: chatPath, defaulted: defaultedChat } = await resolveChatPath(args.chatPath, args.newChat);
  const model = args.model ?? defaultModel;

  if (defaultedChat) console.error(`Using chat file: ${chatPath}`);
  if (args.model === undefined) console.error(`Using model: ${model}`);

  const openrouter = createOpenRouter({ apiKey: Bun.env.OPENROUTER_API_KEY });
  const messages = await loadMessages(chatPath, args.prompt);
  const result = streamText({
    model: openrouter(model),
    system: `This chat lives at ${path.join(process.cwd(), chatPath)}. Read files before editing, they might have changed.`,
    messages,
    tools: makeTools(),
    stopWhen: stepCountIs(100),
    onError: console.error,
  });

  for await (const part of result.fullStream) {
    if (part.type === "text-delta") process.stdout.write(part.text);
    
    if (part.type === "reasoning-start") process.stderr.write("\n<thinking>");
    if (part.type === "reasoning-delta") process.stderr.write(part.text);
    if (part.type === "reasoning-end") process.stderr.write("</thinking>\n");

    if (part.type === "tool-call" && part.toolName === 'useShell') 
      process.stderr.write(`\n<tool> $ ${(part.input as {command: string}).command} </tool>`);
    if (part.type === "tool-result") {
      const {stdout, stderr, exitCode} = (part.output as {stdout: string; stderr: string; exitCode: number})
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
    else if ("listModels" in args) await listModels(args.filter);
    else await runPrompt(args);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.error(error instanceof UsageError ? `${message}\n\n${helpText}` : message);
    process.exit(1);
  }
}
