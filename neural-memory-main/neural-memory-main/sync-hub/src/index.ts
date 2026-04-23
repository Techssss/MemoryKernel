/**
 * Neural Memory Sync Hub — Cloudflare Worker entry point.
 *
 * Dumb ordered log: append incoming changes, return unseen changes.
 * No conflict resolution on hub — client handles it via merge_change_lists().
 */

import { Hono } from "hono";
import { cors } from "hono/cors";
import type { AppEnv } from "./types.js";
import sync from "./routes/sync.js";
import register from "./routes/register.js";
import status from "./routes/status.js";
import devices from "./routes/devices.js";
import health from "./routes/health.js";
import auth from "./routes/auth.js";
import activate from "./routes/activate.js";
import license from "./routes/license.js";
import store from "./routes/store.js";
import { requireAuth } from "./middleware/auth.js";
import { attachLicense } from "./middleware/license.js";
import { handleError } from "./errors.js";

const app = new Hono<AppEnv>();

// --- CORS ---
app.use(
  "/v1/*",
  cors({
    origin: [
      "https://neuralmemory.dev",
      "https://synchub.theio.vn",
      "http://localhost:3000",
      "http://localhost:8000",
    ],
    allowMethods: ["GET", "POST", "DELETE", "OPTIONS"],
    allowHeaders: ["Content-Type", "Authorization"],
    maxAge: 86400,
  }),
);

// --- Global error handler ---
app.onError((err, c) => {
  return handleError(c, err);
});

// --- Public routes ---
app.route("/v1/health", health);
app.route("/v1/auth", auth);

// --- Brain Store ---
// Public: browse, brain/:name, ratings/:name (no auth)
// Protected: publish, rate (auth required, applied in store routes)
app.use("/v1/store/publish", requireAuth);
app.use("/v1/store/rate/*", requireAuth);
app.route("/v1/store", store);

// --- Protected routes (require API key + license check) ---
app.use("/v1/hub/*", requireAuth);
app.use("/v1/hub/*", attachLicense);
app.route("/v1/hub/sync", sync);
app.route("/v1/hub/register", register);
app.route("/v1/hub/status", status);
app.route("/v1/hub/devices", devices);
app.route("/v1/hub/activate", activate);
app.route("/v1/hub/license", license);

// --- Root ---
app.get("/", (c) => {
  return c.json({
    name: "Neural Memory Sync Hub",
    version: "1.0.0",
    docs: "https://nhadaututtheky.github.io/neural-memory",
  });
});

// --- 404 ---
app.notFound((c) => {
  return c.json({ error: "Not found" }, 404);
});

export default app;
