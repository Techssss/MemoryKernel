const axios = require('axios');

jest.mock('axios', () => ({
  create: jest.fn()
}));

const { MemoryKernel } = require('../dist');

function metadata(generation = 1) {
  return {
    workspace_id: 'ws-test',
    generation,
    cache_hit: false,
    degraded: false,
    timestamp: '2026-04-28T00:00:00Z'
  };
}

describe('MemoryKernel Node SDK', () => {
  let http;
  let warnSpy;

  beforeEach(() => {
    http = {
      get: jest.fn().mockResolvedValue({ data: { status: 'ok' } }),
      post: jest.fn(),
      interceptors: {
        response: {
          use: jest.fn()
        }
      }
    };
    axios.create.mockReturnValue(http);
    warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
  });

  afterEach(() => {
    jest.clearAllMocks();
    warnSpy.mockRestore();
  });

  it('configures bearer auth when apiToken is provided', () => {
    new MemoryKernel({
      daemonUrl: 'http://daemon',
      workspaceId: 'ws-test',
      apiToken: 'secret-token'
    });

    expect(axios.create).toHaveBeenCalledWith(expect.objectContaining({
      baseURL: 'http://daemon',
      headers: expect.objectContaining({
        Authorization: 'Bearer secret-token'
      })
    }));
  });

  it('remembers content and updates generation', async () => {
    http.post.mockResolvedValue({
      data: {
        data: { id: 'mem-1' },
        metadata: metadata(7)
      }
    });

    const client = new MemoryKernel({ daemonUrl: 'http://daemon', workspaceId: 'ws-test' });
    const id = await client.remember('User prefers TypeScript', {
      importance: 0.8,
      confidence: 0.9
    });

    expect(id).toBe('mem-1');
    expect(client.generation).toBe(7);
    expect(http.post).toHaveBeenCalledWith('/v1/remember', {
      content: 'User prefers TypeScript',
      importance: 0.8,
      confidence: 0.9,
      workspace_id: 'ws-test'
    });
  });

  it('sends client generation on search', async () => {
    http.post
      .mockResolvedValueOnce({
        data: {
          data: { id: 'mem-1' },
          metadata: metadata(3)
        }
      })
      .mockResolvedValueOnce({
        data: {
          data: { results: [] },
          metadata: metadata(4)
        }
      });

    const client = new MemoryKernel({ daemonUrl: 'http://daemon', workspaceId: 'ws-test' });
    await client.remember('A stored fact');
    const results = await client.search('stored', { limit: 5 });

    expect(results).toEqual([]);
    expect(client.generation).toBe(4);
    expect(http.post).toHaveBeenLastCalledWith('/v1/search', {
      query: 'stored',
      limit: 5,
      workspace_id: 'ws-test',
      client_generation: 3
    });
  });
});
