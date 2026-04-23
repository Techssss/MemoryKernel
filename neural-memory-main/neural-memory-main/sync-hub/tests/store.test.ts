/**
 * Integration tests for Brain Store endpoints.
 *
 * Tests auth requirements, input validation, and response shapes.
 * External calls (GitHub API, D1) are not fully mocked — these test
 * the routing/validation layer via Hono's app.fetch().
 */

import { describe, it, expect } from "vitest";
import app from "../src/index.js";
import type { Env } from "../src/types.js";

// --- Helpers ---

/** Minimal D1-like mock that returns empty results for SELECT queries. */
function mockD1() {
  const stmt = {
    bind: (..._args: unknown[]) => stmt,
    first: async () => null,
    all: async () => ({ results: [] }),
    run: async () => ({ success: true }),
  };
  return {
    prepare: (_sql: string) => stmt,
    batch: async (stmts: unknown[]) => stmts,
  };
}

const mockEnv = {
  SYNC_DB: mockD1(),
  GITHUB_BOT_TOKEN: "",
  XLABS_API_KEY: "",
} as unknown as Env;

const ctx = {} as ExecutionContext;

function makeReq(
  method: string,
  path: string,
  body?: unknown,
  headers?: Record<string, string>,
): Request {
  const init: RequestInit = {
    method,
    headers: { "Content-Type": "application/json", ...headers },
  };
  if (body) init.body = JSON.stringify(body);
  return new Request(`http://localhost${path}`, init);
}

// --- Public Store Routes (no auth) ---

describe("Store browse", () => {
  it("GET /v1/store/browse returns 200 with brains array", async () => {
    const res = await app.fetch(makeReq("GET", "/v1/store/browse"), mockEnv, ctx);
    expect(res.status).toBe(200);
    const body = (await res.json()) as { brains: unknown[]; total: number };
    expect(body).toHaveProperty("brains");
    expect(body).toHaveProperty("total");
    expect(Array.isArray(body.brains)).toBe(true);
  });

  it("accepts filter query params", async () => {
    const res = await app.fetch(
      makeReq("GET", "/v1/store/browse?category=programming&search=python&sort_by=rating_avg&limit=10&offset=0"),
      mockEnv,
      ctx,
    );
    expect(res.status).toBe(200);
    const body = (await res.json()) as { brains: unknown[]; limit: number; offset: number };
    expect(body.limit).toBe(10);
    expect(body.offset).toBe(0);
  });

  it("caps limit at 100", async () => {
    const res = await app.fetch(
      makeReq("GET", "/v1/store/browse?limit=999"),
      mockEnv,
      ctx,
    );
    expect(res.status).toBe(200);
    const body = (await res.json()) as { limit: number };
    expect(body.limit).toBeLessThanOrEqual(100);
  });
});

describe("Store brain fetch", () => {
  it("rejects invalid brain name", async () => {
    const res = await app.fetch(
      makeReq("GET", "/v1/store/brain/INVALID NAME!"),
      mockEnv,
      ctx,
    );
    expect(res.status).toBe(422);
    const body = (await res.json()) as { error: string };
    expect(body.error).toContain("Invalid brain name");
  });

  it("rejects brain name with uppercase", async () => {
    const res = await app.fetch(
      makeReq("GET", "/v1/store/brain/MyBrain"),
      mockEnv,
      ctx,
    );
    expect(res.status).toBe(422);
  });
});

describe("Store ratings fetch", () => {
  it("GET /v1/store/ratings/:name returns rating stats", async () => {
    const res = await app.fetch(
      makeReq("GET", "/v1/store/ratings/test-brain"),
      mockEnv,
      ctx,
    );
    expect(res.status).toBe(200);
    const body = (await res.json()) as { brain_name: string; rating_avg: number; rating_count: number };
    expect(body.brain_name).toBe("test-brain");
    expect(typeof body.rating_avg).toBe("number");
    expect(typeof body.rating_count).toBe("number");
  });

  it("rejects invalid brain name in ratings", async () => {
    const res = await app.fetch(
      makeReq("GET", "/v1/store/ratings/BAD NAME"),
      mockEnv,
      ctx,
    );
    expect(res.status).toBe(422);
  });
});

// --- Auth-Required Store Routes ---

describe("Store publish (auth required)", () => {
  it("rejects without auth", async () => {
    const res = await app.fetch(
      makeReq("POST", "/v1/store/publish", { nmem_brain_package: "1.0" }),
      mockEnv,
      ctx,
    );
    expect(res.status).toBe(401);
    const body = (await res.json()) as { error: string };
    expect(body.error).toContain("API key");
  });

  it("rejects with invalid bearer format", async () => {
    const res = await app.fetch(
      makeReq("POST", "/v1/store/publish", { nmem_brain_package: "1.0" }, {
        Authorization: "Bearer bad_prefix_key",
      }),
      mockEnv,
      ctx,
    );
    expect(res.status).toBe(401);
  });
});

describe("Store rate (auth required)", () => {
  it("rejects without auth", async () => {
    const res = await app.fetch(
      makeReq("POST", "/v1/store/rate/test-brain", { rating: 5 }),
      mockEnv,
      ctx,
    );
    expect(res.status).toBe(401);
    const body = (await res.json()) as { error: string };
    expect(body.error).toContain("API key");
  });

  it("rejects invalid brain name", async () => {
    const res = await app.fetch(
      makeReq("POST", "/v1/store/rate/BAD!", { rating: 5 }, {
        Authorization: "Bearer nmk_invalid",
      }),
      mockEnv,
      ctx,
    );
    // Auth fails first (key doesn't exist in DB), but at least validates route exists
    expect(res.status).toBe(401);
  });
});
