import {
  AIMessage,
  HumanMessage,
  SystemMessage,
  BaseMessage,
} from '@langchain/core/messages';
import { ChatGroq } from '@langchain/groq';
import { Injectable } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import axios from 'axios';
import { ChatMessageDto } from './dto/chat-message.dto';
import { ChatRequestDto } from './dto/chat-request.dto';
import { Annotation, END, START, StateGraph } from '@langchain/langgraph';

const GraphState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (x, y) => x.concat(y),
    default: () => [],
  }),
  searchResults: Annotation<string>({
    reducer: (x, y) => y ?? x,
    default: () => '',
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
    const workflow = new StateGraph(GraphState)
      .addNode('search', async (state) => {
        const lastMessage = state.messages.at(-1)?.content || '';
        const searchInput = typeof lastMessage === 'string' ? lastMessage : JSON.stringify(lastMessage);

        let searchResults = '';
        try {
          const apiResponse = await axios.get(
            `http://localhost:3001/users/search?q=${encodeURIComponent(searchInput)}`,
          );
          const data = apiResponse.data;
          if (
            !data ||
            (Array.isArray(data) && data.length === 0) ||
            (typeof data === 'object' && Object.keys(data).length === 0) ||
            (typeof data === 'string' && data.trim() === '')
          ) {
            searchResults = 'No user with this username';
          } else {
            searchResults =
              typeof data === 'string' ? data : JSON.stringify(data, null, 2);
          }
        } catch (error) {
          searchResults = `Error fetching user search data: ${error instanceof Error ? error.message : String(error)}`;
        }
        return { searchResults };
      })
      .addNode('llm', async (state) => {
        const llm = this.createModel(state.model || this.defaultModel);
        const messages = [
          new SystemMessage(
            `You are a helpful assistant.
            CRITICAL INSTRUCTIONS:
            - If User Search Results is an empty array ([]), "No user with this username", or contains no matching users, respond exactly:
              no user with this username
            - You must respond ONLY using the information provided in the "User Search Results" below.
            - Do NOT use any of your pre-trained model knowledge, outside knowledge, or general knowledge to answer.
            - If the "User Search Results" do not contain the answer, or if the user's request is unrelated to the search results, respond exactly:
              "I cannot answer this question as the required information is not present in the search results."
            - Return EVERY user object.
            - Return EVERY pet object.
            - Return EVERY field exactly as provided.
            - Use ONLY the data provided.
            - Never create sample users.
            - Never invent data.
          User Search Results:${state.searchResults}`,
          ),
          ...state.messages,
        ];
        const response = await llm.invoke(messages);
        return { messages: [response] };
      })
      .addEdge(START, 'search')
      .addEdge('search', 'llm')
      .addEdge('llm', END);

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

    history.push(...newMessages, responseMessage);
    this.histories.set('default', history);

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
