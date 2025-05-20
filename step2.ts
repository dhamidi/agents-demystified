import Anthropic from "@anthropic-ai/sdk";
import type { ContentBlock } from "@anthropic-ai/sdk/resources";

const anthropic = new Anthropic();

type UserInputFn = () => Promise<string>;

function getUserInputFromStdin(): UserInputFn {
  const iterator = console[Symbol.asyncIterator]();
  return async (): Promise<string> => (await iterator.next()).value;
}

class Agent {
  private conversation: Array<Anthropic.MessageParam>;

  constructor(
    private readonly getUserInput: UserInputFn,
    private readonly llm: Anthropic,
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
      });
      for (const content of response.content) {
        await this.showContent(content);
      }
      this.appendToConversation(response.content);
    }
  }

  async showContent(content: Anthropic.ContentBlock): Promise<void> {
    if (content.type !== "text") {
      return;
    }
    process.stdout.write("\x1b[32mClaude\x1b[0m: ");
    process.stdout.write(content.text);
    process.stdout.write("\n");
  }

  appendToConversation(content: ContentBlock[]) {
    this.conversation.push({
      role: "assistant",
      content: content,
    });
  }
}

const agent = new Agent(getUserInputFromStdin(), anthropic);
await agent.run();
