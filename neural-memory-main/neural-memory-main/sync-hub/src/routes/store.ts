/**
 * Brain Store routes — thin proxy for GitHub-based community store.
 *
 * Brain packages are stored on GitHub (unlimited free storage).
 * Hub handles: publish (creates PR via GitHub API) + ratings (D1, tiny data).
 *
 * Public routes: GET /browse, GET /brain/:name, GET /ratings/:name
 * Auth routes: POST /publish, POST /rate/:name
 */

import { Hono } from "hono";
import type { AppEnv } from "../types.js";
import { HubError } from "../errors.js";

const store = new Hono<AppEnv>();

// ── Config ──────────────────────────────────────────────

const GITHUB_REPO = "nhadaututtheky/brain-store";
const GITHUB_BRANCH = "main";
const RAW_BASE = `https://raw.githubusercontent.com/${GITHUB_REPO}/${GITHUB_BRANCH}`;
const API_BASE = `https://api.github.com/repos/${GITHUB_REPO}`;
const MAX_PACKAGE_BYTES = 10_485_760; // 10MB
const MAX_BROWSE_LIMIT = 100;
const INDEX_CACHE_TTL = 300_000; // 5 min in ms
const NAME_RE = /^[a-z0-9][a-z0-9_-]{0,62}[a-z0-9]$/;

// ── In-memory index cache ───────────────────────────────

let indexCache: { data: Record<string, unknown>[]; fetchedAt: number } | null = null;

async function fetchIndex(): Promise<Record<string, unknown>[]> {
  if (indexCache && Date.now() - indexCache.fetchedAt < INDEX_CACHE_TTL) {
    return indexCache.data;
  }

  const resp = await fetch(`${RAW_BASE}/index.json`, {
    headers: { "User-Agent": "neural-memory-hub/1.0" },
  });

  if (!resp.ok) {
    // Return stale cache if available
    if (indexCache) return indexCache.data;
    return [];
  }

  const data = (await resp.json()) as Record<string, unknown>[];
  indexCache = { data, fetchedAt: Date.now() };
  return data;
}

// ── Helpers ─────────────────────────────────────────────

function slugify(name: string): string {
  return name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 64);
}

// ── GET /browse — Browse published brains (from GitHub index) ──

store.get("/browse", async (c) => {
  const category = c.req.query("category") || null;
  const search = c.req.query("search")?.toLowerCase() || null;
  const tag = c.req.query("tag")?.toLowerCase() || null;
  const sortBy = c.req.query("sort_by") || "created_at";
  const rawLimit = parseInt(c.req.query("limit") || "50", 10);
  const rawOffset = parseInt(c.req.query("offset") || "0", 10);
  const limit = Math.min(Number.isNaN(rawLimit) ? 50 : rawLimit, MAX_BROWSE_LIMIT);
  const offset = Number.isNaN(rawOffset) ? 0 : Math.max(0, rawOffset);

  let manifests = await fetchIndex();

  // Filter
  if (category) {
    manifests = manifests.filter((m) => m.category === category);
  }
  if (search) {
    manifests = manifests.filter((m) => {
      const text = `${m.display_name} ${m.description} ${m.author} ${(m.tags as string[] || []).join(" ")}`.toLowerCase();
      return text.includes(search);
    });
  }
  if (tag) {
    manifests = manifests.filter((m) =>
      (m.tags as string[] || []).some((t: string) => t.toLowerCase() === tag),
    );
  }

  // Sort
  if (sortBy === "rating_avg") {
    manifests.sort((a, b) => ((b.rating_avg as number) || 0) - ((a.rating_avg as number) || 0));
  } else if (sortBy === "download_count") {
    manifests.sort((a, b) => ((b.download_count as number) || 0) - ((a.download_count as number) || 0));
  } else {
    manifests.sort((a, b) => ((b.created_at as string) || "").localeCompare((a.created_at as string) || ""));
  }

  // Enrich with hub ratings
  const db = c.env.SYNC_DB;
  const brainNames = manifests.map((m) => m.name as string);
  const ratingsMap = await getHubRatings(db, brainNames);

  const total = manifests.length;
  const paged = manifests.slice(offset, offset + limit);

  const brains = paged.map((m) => {
    const hubRating = ratingsMap.get(m.name as string);
    return {
      ...m,
      description: ((m.description as string) || "").slice(0, 200),
      tags: ((m.tags as string[]) || []).slice(0, 10),
      // Merge hub ratings if available (overrides index values)
      rating_avg: hubRating?.avg ?? (m.rating_avg as number) ?? 0,
      rating_count: hubRating?.count ?? (m.rating_count as number) ?? 0,
    };
  });

  return c.json({ brains, total, limit, offset });
});

// ── GET /brain/:name — Fetch full brain package from GitHub ──

store.get("/brain/:name", async (c) => {
  const name = c.req.param("name");

  if (!NAME_RE.test(name)) {
    throw new HubError(422, "Invalid brain name");
  }

  const resp = await fetch(`${RAW_BASE}/brains/${name}/brain.json`, {
    headers: { "User-Agent": "neural-memory-hub/1.0" },
  });

  if (!resp.ok) {
    throw new HubError(404, "Brain not found");
  }

  const data = await resp.json();
  return c.json(data);
});

// ── POST /publish — Create PR to brain-store repo ──────

store.post("/publish", async (c) => {
  const auth = c.get("auth");
  const ghToken = c.env.GITHUB_BOT_TOKEN;

  if (!ghToken) {
    throw new HubError(503, "Publishing is not configured");
  }

  // Read body
  const body = await c.req.text();
  if (body.length > MAX_PACKAGE_BYTES) {
    throw new HubError(413, `Package too large (max ${MAX_PACKAGE_BYTES} bytes)`);
  }

  let data: Record<string, unknown>;
  try {
    data = JSON.parse(body);
  } catch {
    throw new HubError(400, "Invalid JSON");
  }

  // Validate .brain format
  if (data.nmem_brain_package !== "1.0") {
    throw new HubError(422, "Invalid package: missing nmem_brain_package version");
  }

  const manifest = data.manifest as Record<string, unknown> | undefined;
  if (!manifest) {
    throw new HubError(422, "Invalid package: missing manifest");
  }

  if (!data.snapshot) {
    throw new HubError(422, "Invalid package: missing snapshot");
  }

  const displayName = manifest.display_name as string;
  const description = manifest.description as string;
  const author = manifest.author as string;

  if (!displayName || !description || !author) {
    throw new HubError(422, "Missing required: display_name, description, author");
  }

  // Security: require scan_summary and block dangerous content
  const scanSummary = manifest.scan_summary as Record<string, unknown> | undefined;
  if (!scanSummary || typeof scanSummary.risk_level !== "string" || typeof scanSummary.safe !== "boolean") {
    throw new HubError(422, "Package must include scan_summary with risk_level and safe fields");
  }
  if (scanSummary.risk_level === "high" || scanSummary.risk_level === "critical") {
    throw new HubError(403, "Brain failed security scan — cannot publish");
  }
  if (!scanSummary.safe) {
    throw new HubError(403, "Brain scan reported unsafe — cannot publish");
  }

  const name = (manifest.name as string) || slugify(displayName);
  if (!NAME_RE.test(name)) {
    throw new HubError(422, "Invalid brain name (lowercase alphanumeric, hyphens, 2-64 chars)");
  }

  // Check if brain already exists in index
  const index = await fetchIndex();
  if (index.some((m) => m.name === name)) {
    throw new HubError(409, `Brain '${name}' already exists in the store`);
  }

  // Create PR via GitHub API
  const branchName = `publish/${name}-${Date.now()}`;
  const commitMessage = `feat: add brain "${displayName}" by ${author}`;

  let branchCreated = false;
  try {
    // 1. Get main branch SHA
    const mainRef = await ghApi(ghToken, "GET", `/git/ref/heads/${GITHUB_BRANCH}`);
    const refObj = mainRef.object as Record<string, unknown>;
    const baseSha = refObj.sha as string;

    // 2. Create branch
    await ghApi(ghToken, "POST", "/git/refs", {
      ref: `refs/heads/${branchName}`,
      sha: baseSha,
    });
    branchCreated = true;

    // 3. Create brain.json file
    const brainContent = btoa(unescape(encodeURIComponent(body)));
    await ghApi(ghToken, "PUT", `/contents/brains/${name}/brain.json`, {
      message: commitMessage,
      content: brainContent,
      branch: branchName,
    });

    // 4. Create PR
    const pr = await ghApi(ghToken, "POST", "/pulls", {
      title: commitMessage,
      head: branchName,
      base: GITHUB_BRANCH,
      body: `## New Brain: ${displayName}\n\n` +
        `**Author:** ${author}\n` +
        `**Category:** ${manifest.category || "general"}\n` +
        `**Neurons:** ${(manifest.stats as Record<string, unknown>)?.neuron_count || "?"}\n` +
        `**Size:** ${manifest.size_tier || "?"}\n` +
        `**Tags:** ${((manifest.tags as string[]) || []).join(", ")}\n\n` +
        `> Submitted via Neural Memory Brain Store by user \`${auth.userId}\`\n\n` +
        `---\n_This PR was auto-generated. A GitHub Action will validate the brain package._`,
    });

    return c.json({
      status: "submitted",
      name,
      display_name: displayName,
      pr_number: pr.number,
      pr_url: pr.html_url,
      message: "Brain submitted for review. It will appear in the store after validation.",
    }, 201);
  } catch (err) {
    // Cleanup orphan branch if PR creation failed
    if (branchCreated) {
      try {
        await ghApi(ghToken, "DELETE", `/git/refs/heads/${branchName}`);
      } catch {
        console.error("Failed to cleanup orphan branch:", branchName);
      }
    }
    console.error("GitHub API error:", err);
    throw new HubError(502, "Failed to submit brain to store");
  }
});

// ── POST /rate/:name — Submit a rating (stored in D1) ──

store.post("/rate/:name", async (c) => {
  const auth = c.get("auth");
  const db = c.env.SYNC_DB;
  const brainName = c.req.param("name");

  if (!NAME_RE.test(brainName)) {
    throw new HubError(422, "Invalid brain name");
  }

  const body = await c.req.json<{ rating: number; comment?: string }>();

  if (!body.rating || body.rating < 1 || body.rating > 5 || !Number.isInteger(body.rating)) {
    throw new HubError(422, "Rating must be an integer between 1 and 5");
  }

  const comment = (body.comment || "").slice(0, 500);
  const ratingId = crypto.randomUUID();

  // Upsert rating
  await db
    .prepare(
      `INSERT INTO brain_ratings (id, brain_name, user_id, rating, comment)
       VALUES (?, ?, ?, ?, ?)
       ON CONFLICT(brain_name, user_id) DO UPDATE SET
         rating = excluded.rating,
         comment = excluded.comment,
         created_at = datetime('now')`,
    )
    .bind(ratingId, brainName, auth.userId, body.rating, comment)
    .run();

  // Get updated stats
  const stats = await db
    .prepare(
      "SELECT AVG(rating) as avg, COUNT(*) as count FROM brain_ratings WHERE brain_name = ?",
    )
    .bind(brainName)
    .first<{ avg: number; count: number }>();

  return c.json({
    brain_name: brainName,
    rating_avg: Math.round((stats?.avg || 0) * 100) / 100,
    rating_count: stats?.count || 0,
  });
});

// ── GET /ratings/:name — Get ratings for a brain ────────

store.get("/ratings/:name", async (c) => {
  const db = c.env.SYNC_DB;
  const brainName = c.req.param("name");

  if (!NAME_RE.test(brainName)) {
    throw new HubError(422, "Invalid brain name");
  }

  const stats = await db
    .prepare(
      "SELECT AVG(rating) as avg, COUNT(*) as count FROM brain_ratings WHERE brain_name = ?",
    )
    .bind(brainName)
    .first<{ avg: number; count: number }>();

  const ratings = await db
    .prepare(
      "SELECT rating, comment, created_at FROM brain_ratings WHERE brain_name = ? ORDER BY created_at DESC LIMIT 50",
    )
    .bind(brainName)
    .all();

  return c.json({
    brain_name: brainName,
    rating_avg: Math.round((stats?.avg || 0) * 100) / 100,
    rating_count: stats?.count || 0,
    ratings: ratings.results || [],
  });
});

// ── GitHub API helper ───────────────────────────────────

async function ghApi(
  token: string,
  method: string,
  path: string,
  body?: Record<string, unknown>,
): Promise<Record<string, unknown>> {
  // Always prepend API_BASE — never allow arbitrary URLs
  const url = `${API_BASE}${path}`;
  const resp = await fetch(url, {
    method,
    headers: {
      Authorization: `Bearer ${token}`,
      Accept: "application/vnd.github+json",
      "User-Agent": "neural-memory-hub/1.0",
      "X-GitHub-Api-Version": "2022-11-28",
      ...(body ? { "Content-Type": "application/json" } : {}),
    },
    ...(body ? { body: JSON.stringify(body) } : {}),
  });

  if (!resp.ok) {
    const text = await resp.text();
    console.error(`GitHub API ${method} ${path}: ${resp.status} ${text}`);
    throw new Error(`GitHub API error: ${resp.status}`);
  }

  return (await resp.json()) as Record<string, unknown>;
}

// ── Hub ratings lookup (for enriching browse results) ───

async function getHubRatings(
  db: D1Database,
  brainNames: string[],
): Promise<Map<string, { avg: number; count: number }>> {
  const result = new Map<string, { avg: number; count: number }>();
  if (brainNames.length === 0) return result;

  // D1 doesn't support IN with bindings easily, batch query
  const placeholders = brainNames.map(() => "?").join(",");
  const rows = await db
    .prepare(
      `SELECT brain_name, AVG(rating) as avg, COUNT(*) as count
       FROM brain_ratings
       WHERE brain_name IN (${placeholders})
       GROUP BY brain_name`,
    )
    .bind(...brainNames)
    .all();

  for (const row of rows.results || []) {
    result.set(row.brain_name as string, {
      avg: Math.round((row.avg as number) * 100) / 100,
      count: row.count as number,
    });
  }

  return result;
}

export default store;
