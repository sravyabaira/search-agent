import * as dotenv from 'dotenv';
import { Annotation, END, START, StateGraph } from '@langchain/langgraph';
import { ToolNode } from '@langchain/langgraph/prebuilt';
import { tool } from '@langchain/core/tools';
import { z } from 'zod';
import { ChatGroq } from '@langchain/groq';
import { AIMessage, HumanMessage, SystemMessage, BaseMessage, ToolMessage } from '@langchain/core/messages';

dotenv.config();

let apiCallCount = 0;

const userSearchTool = tool(
  async ({ query }) => {
    apiCallCount++;
    console.log(`[Tool] user_search called with query: "${query}" (Total API calls: ${apiCallCount})`);
    if (query.toLowerCase().includes('john')) {
      return JSON.stringify({
        username: 'john',
        email: 'john@example.com',
        pets: [{ name: 'Fido', species: 'dog' }]
      });
    }
    return 'No user with this username';
  },
  {
    name: 'user_search',
    description: 'Search for users by username or search query to find their details like profiles, names, pets, etc.',
    schema: z.object({
      query: z.string().describe('The username or search query to find user details.'),
    }),
  }
);

const GraphState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (x, y) => x.concat(y),
    default: () => [],
  }),
});

const toolNode = new ToolNode([userSearchTool]);

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

// System prompt we want to test
const systemPrompt = `You are a helpful assistant.
    CRITICAL INSTRUCTIONS:
    - If the user asks about any user, pet, or search query, you MUST use the user_search tool to find the details.
    - Check the message history. If the details for the user, pet, or search query being asked about are already present in the tool results in the message history, do NOT call the user_search tool again. Use the existing results in the history to answer.
    - Only call the user_search tool if the requested information is not present in any previous tool results in the history.
    - If the user_search tool returns "No user with this username" or empty results, respond exactly:
      no user with this username
    - You must respond ONLY using the information provided in the tool results.
    - Do NOT use any of your pre-trained model knowledge, outside knowledge, or general knowledge to answer.
    - If the tool results do not contain the answer, or if the user's request is unrelated to the tool results, respond exactly:
      "I cannot answer this question as the required information is not present in the search results."
    - Return EVERY user object.
    - Return EVERY pet object.
    - Return EVERY field exactly as provided.
    - Use ONLY the data provided.
    - Never create sample users.
    - Never invent data.`;

const workflow = new StateGraph(GraphState)
  .addNode('agent', async (state) => {
    const llm = new ChatGroq({
      model: process.env.GROQ_MODEL || 'llama-3.1-8b-instant',
      apiKey: process.env.GROQ_API_KEY,
      temperature: 0,
    });
    
    const lastMessage = state.messages.at(-1);
    const hasToolMessage = lastMessage instanceof ToolMessage;
    const llmWithTools = hasToolMessage ? llm : llm.bindTools([userSearchTool]);

    const messages = [new SystemMessage(systemPrompt), ...state.messages];
    const response = await llmWithTools.invoke(messages);
    return { messages: [response] };
  })
  .addNode('tools', toolNode)
  .addEdge(START, 'agent')
  .addConditionalEdges('agent', shouldContinue)
  .addEdge('tools', 'agent');

const app = workflow.compile();

async function run() {
  try {
    let history: BaseMessage[] = [];

    console.log('\n--- Turn 1: Find user john ---');
    let result = await app.invoke({
      messages: [...history, new HumanMessage('Find user john')],
    });
    history = result.messages;
    console.log('Assistant response:', history.at(-1)?.content);

    console.log('\n--- Turn 2: What is his email? (Follow up) ---');
    result = await app.invoke({
      messages: [...history, new HumanMessage('What is his email?')],
    });
    history = result.messages;
    console.log('Assistant response:', history.at(-1)?.content);

    console.log('\n--- Turn 3: Find user mary (New query) ---');
    result = await app.invoke({
      messages: [...history, new HumanMessage('Find user mary')],
    });
    history = result.messages;
    console.log('Assistant response:', history.at(-1)?.content);

    console.log('\nTotal API calls made:', apiCallCount);
  } catch (err) {
    console.error('Error:', err);
  }
}

run();
