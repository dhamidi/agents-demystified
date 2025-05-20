import Anthropic from "@anthropic-ai/sdk";
import type { ContentBlockParam } from "@anthropic-ai/sdk/resources";
import type { ToolUseBlockParam } from "@anthropic-ai/sdk/resources.js";

const anthropic = new Anthropic();

type UserInputFn = () => Promise<string>;

function getUserInputFromStdin(): UserInputFn {
  const iterator = console[Symbol.asyncIterator]();
  return async (): Promise<string> => (await iterator.next()).value;
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
    process.stdout.write(
      `[${content.tool_use_id}] ${JSON.stringify(content.content, null, 2)}`,
    );
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
        model: "claude-3-7-sonnet-20250219",
        max_tokens: 1024,
        messages: this.conversation,
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

class ToolBox {
  constructor(private readonly tools: Tool[] = []) {}

  find(name: string): Tool | undefined {
    return this.tools.find((tool) => tool.name === name);
  }
  toParam(): Anthropic.Tool[] {
    return this.tools.map((tool) => tool.toParam());
  }
}

const toolbox = new ToolBox([new RunTerminalTool()]);
const agent = new Agent(
  getUserInputFromStdin(),
  anthropic,
  toolbox,
  new StdoutDisplay(),
);
await agent.run();
