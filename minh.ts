#!/usr/bin/env bun
import { createOpenRouter } from "@openrouter/ai-sdk-provider";
import { OpenRouter } from "@openrouter/sdk";
import { addToolInputExamplesMiddleware, stepCountIs, streamText, tool, wrapLanguageModel, type LanguageModelUsage, type ModelMessage } from "ai";
import { appendFile, readdir, writeFile } from "node:fs/promises";
import { parseArgs } from "node:util";
import readline from "node:readline";
import * as z from 'zod';
import type { WriteStream } from "node:tty";
import path from "node:path";

const defaultModel = "openrouter/free";
class UsageError extends Error {}

const shellOutputLimit = 10_000;
const tailText = (text: string) =>
  text.length > shellOutputLimit ? `truncated, last ${shellOutputLimit} chars:\n${text.slice(-shellOutputLimit)}` : text;

const formatTokens = (tokens: number | undefined) =>
  tokens === undefined ? "?" : tokens.toLocaleString("en-US");

function formatUsage(usage: LanguageModelUsage) {
  const inputDetails = usage.inputTokenDetails;
  const outputDetails = usage.outputTokenDetails;
  const cacheReadTokens = inputDetails.cacheReadTokens ?? usage.cachedInputTokens;
  const reasoningTokens = outputDetails.reasoningTokens ?? usage.reasoningTokens;

  const inputBreakdown = [
    inputDetails.noCacheTokens === undefined ? undefined : `${formatTokens(inputDetails.noCacheTokens)} new`,
    cacheReadTokens === undefined ? undefined : `${formatTokens(cacheReadTokens)} cached read`,
    inputDetails.cacheWriteTokens === undefined ? undefined : `${formatTokens(inputDetails.cacheWriteTokens)} cached write`,
  ].filter(Boolean).join(", ");

  const outputBreakdown = [
    outputDetails.textTokens === undefined ? undefined : `${formatTokens(outputDetails.textTokens)} text`,
    reasoningTokens === undefined ? undefined : `${formatTokens(reasoningTokens)} reasoning`,
  ].filter(Boolean).join(", ");

  return `Usage: ${formatTokens(usage.totalTokens)} tokens\n  input:  ${formatTokens(usage.inputTokens)}${inputBreakdown ? ` (${inputBreakdown})` : ""}\n  output: ${formatTokens(usage.outputTokens)}${outputBreakdown ? ` (${outputBreakdown})` : ""}`;
}

const helpText = `Usage:
  minh [-y] [--new-chat] [-c <chat-file>] [-m <model>] <prompt>
  minh --list-models [filter]
Options:
  -c, --chat <path>  Path to the .chat file
  -m, --model <id>   OpenRouter model id (default: openrouter/free)
  -y, --yes          Run tools without asking for permission
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
      yes: { type: "boolean", short: "y" },
      "list-models": { type: "boolean" },
      "new-chat": { type: "boolean" },
    },
    allowPositionals: true,
  });
  const { values, positionals } = parsed;
  const { chat, help, model, yes, "list-models": listModels, "new-chat": newChat } = values;
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

  return { chatPath: chat, model, prompt, yes, newChat };
}

const timestampChat = /^(\d+)\.chat$/;

function chatTimestamp(file: string) {
  const match = timestampChat.exec(file);
  return match ? Number.parseInt(match[1]!, 10) : undefined;
}

async function createTimestampChat() {
  const next = `./${Date.now()}.chat`;
  await writeFile(next, "", { flag: "wx" });
  return { path: next, defaulted: true };
}

async function resolveChatPath(chatPath?: string, newChat = false) {
  if (chatPath) return { path: chatPath, defaulted: false };
  if (newChat) return createTimestampChat();

  const latest = (await readdir("."))
    .map(file => ({ file, timestamp: chatTimestamp(file) }))
    .filter((entry): entry is { file: string; timestamp: number } => entry.timestamp !== undefined)
    .sort((a, b) => b.timestamp - a.timestamp || b.file.localeCompare(a.file))[0];

  if (latest) return { path: `./${latest.file}`, defaulted: true };

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
  const lines = (await Bun.file(chatPath).text()).split("\n").filter(line => line.trim());
  const messages: ModelMessage[] = [];

  for (const line of lines)
    messages.push(JSON.parse(line));

  messages.push({ role: "user", content: prompt });
  return { messages };
}

async function appendMessages(chatPath: string, messages: ModelMessage[]) {
  await appendFile(chatPath, messages.map(message => JSON.stringify(message)).join("\n") + "\n");
}

function makeTools(yes = false) { return {
  useShell: tool({
    description: "Run a shell command in the current working directory and return stdout, stderr, and exit code.",
    inputSchema: z.object({
      command: z.string().describe("Shell command to run."),
    }),
    inputExamples: [
      {
        input: {
          command: `patch -p0 <<'PATCH'
--- index.ts
+++ index.ts
@@ -1,3 +1,3 @@
-old text
+new text
PATCH`,
        },
      },
      {
        input: {
          command: `# List executable commands available on PATH, grouped by directory.
printf '%s\\n' "$PATH" | tr ':' '\\n' | awk 'NF && !seen[$0]++' |
while IFS= read -r d; do
  [ -d "$d" ] || continue
  printf '\\n%s:\\n' "$d"
  for f in "$d"/*; do
    [ -f "$f" ] && [ -x "$f" ] && basename "$f"
  done | sort | sed 's/^/  /'
done`,
        },
      },
    ],
    execute: async ({ command }) => {
      if (!yes) {
        await new Promise<void>((resolve) => setTimeout(() => resolve(), 1)) // wait for stdout to flush
        const rl = readline.promises.createInterface(process.stdin, process.stdout);
        const answer = await rl.question(`Can I run \`${command}\` (Y/n)? `);
        rl.close()
        if (answer !== 'Y') throw new Error('Rejected by User');
      }
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
}; }

async function runPrompt(args: { chatPath?: string; model?: string; prompt: string; yes?: boolean; newChat?: boolean }) {
  const { path: chatPath, defaulted: defaultedChat } = await resolveChatPath(args.chatPath, args.newChat);
  const model = args.model ?? defaultModel;

  if (defaultedChat) console.error(`Using chat file: ${chatPath}`);
  if (args.model === undefined) console.error(`Using model: ${model}`);

  const chat = await loadMessages(chatPath, args.prompt);
  const openrouter = createOpenRouter({ apiKey: Bun.env.OPENROUTER_API_KEY });
  const result = streamText({
    model: wrapLanguageModel({
      model: openrouter(model),
      middleware: addToolInputExamplesMiddleware(),
    }),
    system: `This chat lives at ${path.join(process.cwd(), chatPath)}. Read files before editing.`,
    ...chat,
    tools: makeTools(args.yes),
    stopWhen: stepCountIs(100),
    onError: () => {},
  });

  const print = (text: string, p: WriteStream = process.stdout) => p.write(text);

  for await (const part of result.fullStream) {
    if (part.type === "text-delta") print(part.text);
    
    if (part.type === "reasoning-start") print("\n<thinking>", process.stderr);
    if (part.type === "reasoning-delta") print(part.text, process.stderr);
    if (part.type === "reasoning-end") print("</thinking>\n", process.stderr);

    if (part.type === "tool-call" && part.toolName === 'useShell') 
      print(`\n<tool> $ ${(part.input as {command: string}).command} </tool>`, process.stderr);
    if (part.type === "tool-result") {
      const {stdout, stderr, exitCode} = (part.output as {stdout: string; stderr: string; exitCode: number})
      if (stderr) print(`\n<tool-error>Exit ${exitCode}: ${stderr}</tool-error>`, process.stderr)
      if (stdout) print(`\n<tool-result>${stdout}</tool-result>\n`, process.stderr);
    }

    if (part.type === "tool-error")
      console.error(`\nTool ${part.toolName} failed: ${String(part.error)}`);
    if (part.type === "error")
      throw part.error instanceof Error ? part.error : new Error(String(part.error));
    if (part.type === "abort")
      throw new Error(part.reason ?? "Stream aborted");
  }
  process.stdout.write("\n");
  const [response, totalUsage] = await Promise.all([result.response, result.totalUsage]);
  console.error(formatUsage(totalUsage));
  await appendMessages(chatPath, [
    { role: "user", content: args.prompt },
    ...response.messages,
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
