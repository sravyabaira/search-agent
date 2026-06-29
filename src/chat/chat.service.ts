import {
  AIMessage,
  HumanMessage,
  SystemMessage,
  BaseMessage,
  ToolMessage,
} from '@langchain/core/messages';
import { ChatGroq } from '@langchain/groq';
import { Injectable } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import axios from 'axios';
import { ChatMessageDto } from './dto/chat-message.dto';
import { ChatRequestDto } from './dto/chat-request.dto';
import { Annotation, END, START, StateGraph } from '@langchain/langgraph';
import { ToolNode } from '@langchain/langgraph/prebuilt';
import { tool } from '@langchain/core/tools';
import { z } from 'zod';

const userSearchTool = tool(
  async ({ query }) => {
    try {
      const apiResponse = await axios.get(
        `http://localhost:3001/users/search?q=${encodeURIComponent(query)}`,
      );
      const data = apiResponse.data;

      // Check for genuinely empty responses
      const isEmpty =
        data === null ||
        data === undefined ||
        (Array.isArray(data) && data.length === 0) ||
        (typeof data === 'string' && data.trim() === '');

      if (isEmpty) {
        return `SEARCH_RESULT for "${query}": NO_USERS_FOUND. No user exists with this username or matching this query.`;
      }

      const formattedData = typeof data === 'string' ? data : JSON.stringify(data, null, 2);
      return `SEARCH_RESULT for "${query}": USERS_FOUND. Data:\n${formattedData}`;
    } catch (error) {
      return `Error fetching user search data: ${error instanceof Error ? error.message : String(error)}`;
    }
  },
  {
    name: 'user_search',
    description: 'Search for users by username or search query to find their details like profiles, names, pets, etc.',
    schema: z.object({
      query: z.string().describe('The username or search query to find user details.'),
    }),
  }
);

const calculationTool = tool(
  async ({ expression }) => {
    try {
      const result = Function(`"use strict"; return (${expression})`)();
      return `${expression} = ${result}`;
    } catch (error) {
      return `Error calculating: ${error instanceof Error ? error.message : String(error)}`;
    }
  },
  {
    name: 'calculation_tool',
    description: 'Evaluates a math expression and returns the result.',
    schema: z.object({
      expression: z.string().describe('A math expression to evaluate, e.g. "2 + 2" or "10 * 5 / 2"'),
    }),
  },
);

const tools = [userSearchTool, calculationTool];
const toolNode = new ToolNode(tools);

const GraphState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (x, y) => x.concat(y),
    default: () => [],
  }),
  model: Annotation<string>({
    reducer: (x, y) => y ?? x,
    default: () => '',
  }),
});

@Injectable()
export class ChatService {
  private readonly defaultModel: string;
  private histories = new Map<string, any[]>();
  private readonly app: any;

  constructor(private readonly configService: ConfigService) {
    this.defaultModel =
      this.configService.get<string>('GROQ_MODEL') ?? 'llama-3.1-8b-instant';
    this.app = this.initializeGraph();
  }

  private initializeGraph() {

    const shouldContinue = (state: typeof GraphState.State) => {
      const lastMessage = state.messages.at(-1);
      if (
        lastMessage &&
        'tool_calls' in lastMessage &&
        Array.isArray(lastMessage.tool_calls) &&
        lastMessage.tool_calls.length > 0
      ) {
        return 'tools';
      }
      return END;
    };

    const workflow = new StateGraph(GraphState)
      .addNode('agent', async (state) => {
        const llm = this.createModel(state.model || this.defaultModel);
        const lastMessage = state.messages.at(-1);
        const hasToolMessage = lastMessage instanceof ToolMessage;
        const llmWithTools = hasToolMessage ? llm : llm.bindTools(tools);

        const systemPrompt = `You are a helpful assistant that searches for user information.

          When the user asks about a person, pet, or anything that requires looking up user data:
          1. First check the conversation history for previous tool results that already contain the answer. If found, reuse that data without calling the tool again.
          2. If the information is NOT already in the history, call the user_search tool.

          How to interpret tool results:
          - If the result contains "USERS_FOUND", the search was successful. Present ALL the user data returned — every field, every user object, every pet object — exactly as provided.
          - If the result contains "NO_USERS_FOUND", respond with exactly: "no user with this username"

          Strict rules:
          - Base your answers ONLY on data from tool results. Never invent, fabricate, or assume any user data.
          - Never use your pre-trained knowledge to answer questions about users.
          - If the user asks something unrelated to user searches, or the tool results don't contain the answer, respond with: "I cannot answer this question as the required information is not present in the search results."
          - Present all data exactly as returned — do not omit fields, users, or pets.`;

        const messages = [new SystemMessage(systemPrompt), ...state.messages];
        const response = await llmWithTools.invoke(messages);
        return { messages: [response] };
      })
      .addNode('weatherNode', async (state) => {
        const lastMsg = state.messages.at(-1);
        const content = lastMsg?.content as string;

        // Has to guess — no city in "search for sravya"
        const cityMatch = content.match(/weather in (\w+)/i);
        if (!cityMatch) {
          console.log('[Weather] No city found, skipping');
          return {}; // exits silently
        }

        const city = cityMatch[1];
        // fetch weather...
        return { messages: [new AIMessage(`Weather in ${city}: 28°C`)] };
      })

      .addNode('tools', toolNode)
      .addEdge(START, 'agent')
      .addEdge(START, 'weatherNode')
      .addConditionalEdges('agent', shouldContinue)
      .addEdge('tools', 'agent');

    return workflow.compile();
  }

  private createModel(model: string) {
    return new ChatGroq({
      model,
      apiKey: this.configService.getOrThrow<string>('GROQ_API_KEY'),
      temperature: 0,
    });
  }

  private toLangChainMessages(messages: ChatMessageDto[]) {
    return messages.map((message) => {
      switch (message.role) {
        case 'system':
          return new SystemMessage(message.content);
        case 'assistant':
          return new AIMessage(message.content);
        default:
          return new HumanMessage(message.content);
      }
    });
  }

  async chat(dto: ChatRequestDto) {
    const model = dto.model ?? this.defaultModel;

    let history = this.histories.get('default') || [];
    const newMessages = this.toLangChainMessages(dto.messages);

    const resultState = await this.app.invoke({
      messages: [...history, ...newMessages],
      model,
    });

    const responseMessage = resultState.messages.at(-1) as AIMessage;

    const newlyAddedMessages = resultState.messages.slice(history.length);
    this.histories.set('default', [...history, ...newlyAddedMessages]);

    return {
      model,
      message: {
        role: 'assistant',
        content:
          typeof responseMessage.content === 'string'
            ? responseMessage.content
            : JSON.stringify(responseMessage.content ?? ''),
      },
      usage: (responseMessage.response_metadata as { usage?: unknown } | undefined)
        ?.usage,
    };
  }
}
