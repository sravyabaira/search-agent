import { AIMessage } from '@langchain/core/messages';
import { ConfigService } from '@nestjs/config';
import { Test, TestingModule } from '@nestjs/testing';
import axios from 'axios';
import { ChatService } from './chat.service';

const mockInvoke = jest.fn();

jest.mock('@langchain/groq', () => {
  return {
    ChatGroq: jest.fn().mockImplementation(() => {
      return {
        invoke: mockInvoke,
        bindTools: jest.fn().mockReturnThis(),
      };
    }),
  };
});

jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

describe('ChatService', () => {
  let service: ChatService;

  beforeEach(async () => {
    mockInvoke.mockReset();
    mockInvoke.mockResolvedValue(new AIMessage('Hello from Groq!'));
    mockedAxios.get.mockReset();
    mockedAxios.get.mockResolvedValue({ data: [{ username: 'test', email: 'test@test.com' }] });

    const module: TestingModule = await Test.createTestingModule({
      providers: [
        ChatService,
        {
          provide: ConfigService,
          useValue: {
            get: (key: string) => {
              if (key === 'GROQ_MODEL') return 'llama-3.1-8b-instant';
              return undefined;
            },
            getOrThrow: (key: string) => {
              if (key === 'GROQ_API_KEY') return 'test-api-key';
              throw new Error(`Missing config: ${key}`);
            },
          },
        },
      ],
    }).compile();

    service = module.get(ChatService);
  });

  it('calls search API and returns an assistant message', async () => {
    mockInvoke
      .mockResolvedValueOnce(
        new AIMessage({
          content: '',
          tool_calls: [
            {
              name: 'user_search',
              args: { query: 'Search query' },
              id: 'call_1',
              type: 'tool_call',
            },
          ],
        }),
      )
      .mockResolvedValueOnce(new AIMessage('Hello from Groq!'));

    const result = await service.chat({
      messages: [{ role: 'user', content: 'Search query' }],
    });

    expect(mockedAxios.get).toHaveBeenCalledWith(
      'http://localhost:3001/users/search?q=Search%20query',
    );
    expect(mockInvoke).toHaveBeenCalledTimes(2);
    expect(result).toEqual({
      model: 'llama-3.1-8b-instant',
      message: {
        role: 'assistant',
        content: 'Hello from Groq!',
      },
      usage: undefined,
    });
  });

  it('returns "no user with this username" when API response is empty', async () => {
    mockedAxios.get.mockResolvedValue({ data: [] });
    mockInvoke
      .mockResolvedValueOnce(
        new AIMessage({
          content: '',
          tool_calls: [
            {
              name: 'user_search',
              args: { query: 'NonExistentUser' },
              id: 'call_1',
              type: 'tool_call',
            },
          ],
        }),
      )
      .mockResolvedValueOnce(new AIMessage('no user with this username'));

    const result = await service.chat({
      messages: [{ role: 'user', content: 'NonExistentUser' }],
    });

    expect(mockedAxios.get).toHaveBeenCalledWith(
      'http://localhost:3001/users/search?q=NonExistentUser',
    );
    expect(mockInvoke).toHaveBeenCalledTimes(2);
    expect(result.message.content).toBe('no user with this username');
  });

  it('instructs the LLM to only use search results and not external model knowledge', async () => {
    await service.chat({
      messages: [{ role: 'user', content: 'What is the capital of France?' }],
    });

    expect(mockInvoke).toHaveBeenCalledTimes(1);
    const invokedMessages = mockInvoke.mock.calls[0][0];
    const systemMessage = invokedMessages.find(
      (msg: any) => msg.constructor.name === 'SystemMessage',
    );

    expect(systemMessage).toBeDefined();
    expect(systemMessage.content).toContain(
      'Base your answers ONLY on data from tool results.',
    );
    expect(systemMessage.content).toContain(
      'Never use your pre-trained knowledge to answer questions about users.',
    );
  });
});

