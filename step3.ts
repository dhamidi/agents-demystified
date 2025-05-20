import Anthropic from "@anthropic-ai/sdk";
import type { ContentBlock } from "@anthropic-ai/sdk/resources";

const anthropic = new Anthropic();

type UserInputFn = () => Promise<string>;

function getUserInputFromStdin(): UserInputFn {
  const iterator = console[Symbol.asyncIterator]();
  return async (): Promise<string> => (await iterator.next()).value;
}

interface Display {
  showTextContent(content: Anthropic.TextBlock): Promise<void>;
  showToolUseContent(content: Anthropic.ToolUseBlock): Promise<void>;
}

class StdoutDisplay {
  async showTextContent(content: Anthropic.TextBlock): Promise<void> {
    process.stdout.write("\x1b[32mClaude\x1b[0m: ");
    process.stdout.write(content.text);
    process.stdout.write("\n");
  }
  async showToolUseContent(content: Anthropic.ToolUseBlock): Promise<void> {
    process.stdout.write("\x1b[33mTool\x1b[0m: ");
    process.stdout.write(
      `[${content.id}] ${content.name} ${JSON.stringify(content.input, null, 2)}`,
    );
    process.stdout.write("\n");
  }
}

class Agent {
  private conversation: Array<Anthropic.MessageParam>;

  constructor(
    private readonly getUserInput: UserInputFn,
    private readonly llm: Anthropic,
    private readonly tools: Anthropic.Tool[],
    private readonly display: Display,
  ) {
    this.conversation = [];
  }

  async run(): Promise<never> {
    while (true) {
      process.stdout.write("\x1b[31mYou\x1b[0m: ");
      const inputMessage = await this.getUserInput();
      this.conversation = [
        ...this.conversation,
        {
          role: "user",
          content: inputMessage,
        },
      ];
      const response = await this.llm.messages.create({
        model: "claude-3-7-sonnet-20250219",
        max_tokens: 1024,
        messages: this.conversation,
        tools: this.tools,
      });
      for (const content of response.content) {
        await this.showContent(content);
      }
      this.appendToConversation(response.content);
    }
  }

  async showContent(content: Anthropic.ContentBlock): Promise<void> {
    switch (content.type) {
      case "text":
        return await this.display.showTextContent(content);
      case "tool_use":
        return await this.display.showToolUseContent(content);
      default:
        return;
    }
  }

  appendToConversation(content: ContentBlock[]) {
    this.conversation.push({
      role: "assistant",
      content: content,
    });
  }
}

const toolbox: Anthropic.Tool[] = [
  {
    name: "run_terminal_command",
    description: `Use this tool to run commands on the user's laptop.`,
    input_schema: {
      type: "object",
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
    },
  },
];
const agent = new Agent(
  getUserInputFromStdin(),
  anthropic,
  toolbox,
  new StdoutDisplay(),
);
await agent.run();
