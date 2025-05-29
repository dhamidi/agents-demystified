import fs from "fs";
import Anthropic from "@anthropic-ai/sdk";
import type {
  ContentBlockParam,
  ImageBlockParam,
  TextBlockParam,
} from "@anthropic-ai/sdk/resources";
import type { ToolUseBlockParam } from "@anthropic-ai/sdk/resources.js";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import type { CallToolRequest } from "@modelcontextprotocol/sdk/types.js";
import { parseArgs } from "util";

const anthropic = new Anthropic();

type UserInputFn = () => Promise<string>;

function getUserInputFromStdin(): UserInputFn {
  const iterator = console[Symbol.asyncIterator]();
  return async (): Promise<string> => (await iterator.next()).value;
}

function getUserInputFromFile(
  name: string,
  fallback: UserInputFn,
): UserInputFn {
  const contents = [fs.readFileSync(name).toString()];

  return async (): Promise<string> =>
    new Promise((resolve, reject) => {
      const head = contents.pop();
      if (head) {
        console.log(head);
        resolve(head);
      } else {
        fallback().then(resolve).catch(reject);
      }
    });
}

interface Display {
  showTextContent(content: Anthropic.TextBlockParam): Promise<void>;
  showToolUseContent(content: Anthropic.ToolUseBlockParam): Promise<void>;
  showToolResultContent(content: Anthropic.ToolResultBlockParam): Promise<void>;
}

class StdoutDisplay {
  async showTextContent(content: Anthropic.TextBlockParam): Promise<void> {
    process.stdout.write("\x1b[32mClaude\x1b[0m: ");
    process.stdout.write(content.text);
    process.stdout.write("\n");
  }
  async showToolUseContent(
    content: Anthropic.ToolUseBlockParam,
  ): Promise<void> {
    process.stdout.write("\x1b[33mTool\x1b[0m: ");
    process.stdout.write(
      `[${content.id}] ${content.name} ${JSON.stringify(content.input, null, 2)}`,
    );
    process.stdout.write("\n");
  }
  async showToolResultContent(
    content: Anthropic.ToolResultBlockParam,
  ): Promise<void> {
    process.stdout.write("\x1b[33mTool\x1b[0m: ");
    if (typeof content.content === "string") {
      process.stdout.write(`[${content.tool_use_id}] ${content}`);
    } else if (content.content) {
      for (const block of content.content) {
        if (block.type === "text") {
          this.showTextContent(block);
        }
      }
    }
    process.stdout.write("\n");
  }
}

class Tool {
  get name(): string {
    throw new Error("not implemented");
  }
  get description(): string {
    throw new Error("not implemented");
  }
  get schema(): {
    properties: Record<string, any>;
    required?: string[];
  } {
    throw new Error("not implemented");
  }

  toParam(): Anthropic.Tool {
    return {
      name: this.name,
      description: this.description,
      input_schema: { type: "object", ...this.schema },
    };
  }

  async execute(_content: ToolUseBlockParam): Promise<ContentBlockParam[]> {
    throw new Error("not implemented");
  }
}

class Agent {
  private conversation: Array<Anthropic.MessageParam>;

  constructor(
    private readonly getUserInput: UserInputFn,
    private readonly llm: Anthropic,
    private readonly tools: ToolBox,
    private readonly display: Display,
    private readonly systemPrompt?: string,
  ) {
    this.conversation = [];
  }

  async run(): Promise<never> {
    let needsInput: boolean = true;
    while (true) {
      if (needsInput) {
        process.stdout.write("\x1b[31mYou\x1b[0m: ");
        const inputMessage = await this.getUserInput();
        this.conversation = [
          ...this.conversation,
          {
            role: "user",
            content: inputMessage,
          },
        ];
      }

      needsInput = true;
      const response = await this.llm.messages.create({
        model: "claude-3-5-sonnet-20241022",
        max_tokens: 1024,
        messages: this.conversation,
        system: this.systemPrompt,
        tools: this.tools.toParam(),
      });
      this.appendToConversation(response.role, response.content);
      for (const content of response.content) {
        await this.showContent(content);
        if (content.type === "tool_use") {
          const toolResult = await this.executeTool(content);
          for (const result of toolResult) {
            this.showContent(result);
          }
          this.appendToConversation("user", toolResult);
          needsInput = false;
        }
      }
    }
  }

  async executeTool(
    content: Anthropic.ToolUseBlock,
  ): Promise<ContentBlockParam[]> {
    const tool = this.tools.find(content.name);
    if (!tool) {
      return [
        {
          type: "tool_result",
          tool_use_id: content.id,
          is_error: true,
          content: `Unsupported tool: ${content.name}`,
        },
      ];
    }

    return await tool.execute(content);
  }

  async showContent(content: Anthropic.ContentBlockParam): Promise<void> {
    switch (content.type) {
      case "text":
        return await this.display.showTextContent(content);
      case "tool_use":
        return await this.display.showToolUseContent(content);
      case "tool_result":
        return await this.display.showToolResultContent(content);
      default:
        return;
    }
  }

  appendToConversation(
    role: "assistant" | "user",
    content: ContentBlockParam[],
  ) {
    this.conversation.push({
      role,
      content: content,
    });
  }
}

class RunTerminalTool extends Tool {
  get name(): string {
    return "run_terminal_command";
  }
  get description(): string {
    return `Use this tool to run commands on the user's laptop.`;
  }
  get schema() {
    return {
      properties: {
        cmd: {
          type: "string",
        },
        args: {
          type: "array",
          items: { type: "string" },
        },
      },
      required: ["cmd", "args"],
    };
  }
  async execute(
    content: Anthropic.ToolUseBlockParam,
  ): Promise<ContentBlockParam[]> {
    const input = content.input as {
      cmd: string;
      args: string[];
    };
    const proc = Bun.spawn([input.cmd, ...input.args], { stderr: 1 });
    const stdout = await new Response(proc.stdout).text();
    return [
      {
        type: "tool_result",
        tool_use_id: content.id,
        is_error: proc.exitCode !== 0,
        content: stdout,
      },
    ];
  }
}

class MCPTool extends Tool {
  constructor(
    private readonly client: Client,
    private readonly toolDefinition: {
      name: string;
      description?: string;
      inputSchema: { type: "object" } & Record<string, any>;
    },
  ) {
    super();
  }
  get name(): string {
    return this.toolDefinition.name;
  }
  get description(): string {
    return this.toolDefinition.description ?? "";
  }
  get schema() {
    const toolSchema = this.toolDefinition.inputSchema;
    return {
      properties: toolSchema.properties,
      required: toolSchema.required,
    };
  }
  async execute(content: ToolUseBlockParam): Promise<ContentBlockParam[]> {
    const result = await this.client.callTool({
      name: content.name,
      arguments: content.input as CallToolRequest["params"]["arguments"],
    });
    return [
      {
        type: "tool_result",
        tool_use_id: content.id,
        content: result.content as (TextBlockParam | ImageBlockParam)[],
        is_error: result.isError as boolean | undefined,
      },
    ];
  }
}

class ToolBox {
  constructor(private readonly tools: Tool[] = []) {}

  find(name: string): Tool | undefined {
    return this.tools.find((tool) => tool.name === name);
  }
  toParam(): Anthropic.Tool[] {
    return this.tools.map((tool) => tool.toParam());
  }

  [Symbol.iterator]() {
    return this.tools[Symbol.iterator]();
  }
}

const playwright = new Client({
  name: "step5",
  version: "1.0.0",
});
const transport = new StdioClientTransport({
  command: "npx",
  args: ["@playwright/mcp@latest"],
});
playwright.connect(transport);
const playwrightToolDefinitions = await playwright.listTools();
const playwrightTools = playwrightToolDefinitions.tools.map(
  (toolDef) => new MCPTool(playwright, toolDef),
);
const toolbox = new ToolBox([new RunTerminalTool(), ...playwrightTools]);
console.log("Known tools: ");
for (const tool of toolbox) {
  console.log(tool.name, tool.description);
}

const { values } = parseArgs({
  args: process.argv,
  options: {
    prompt: {
      type: "string",
    },
    system: {
      type: "string",
    },
    "hide-tool-result": {
      type: "boolean",
    },
  },
  allowPositionals: true,
});

const display = new StdoutDisplay();
if (values["hide-tool-result"]) {
  display.showToolResultContent = async () => undefined;
}

const systemPrompt = await getSystemPrompt(values.system);
const agent = new Agent(
  values.prompt
    ? getUserInputFromFile(values.prompt, getUserInputFromStdin())
    : getUserInputFromStdin(),
  anthropic,
  toolbox,
  display,
  systemPrompt,
);

if (systemPrompt) {
  console.log(systemPrompt);
}

await agent.run();

async function getSystemPrompt(filename?: string): Promise<string | undefined> {
  if (!filename) {
    return undefined;
  }
  try {
    return fs.readFileSync(filename).toString("utf8");
  } catch {
    return undefined;
  }
}
