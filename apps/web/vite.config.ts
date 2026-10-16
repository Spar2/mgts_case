import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type { Connect, Plugin, ViteDevServer } from "vite";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const webRoot = path.dirname(fileURLToPath(import.meta.url));

function localAdminFile() {
  return [
    process.env.DTS_LOCAL_ADMIN_FILE,
    path.resolve(webRoot, "../../.runtime/local-admin.json"),
    path.resolve(webRoot, "../../../web-design-delivery/.runtime/local-admin.json"),
  ].find((file) => file && fs.existsSync(file));
}

function loopback(request: Connect.IncomingMessage) {
  const address = request.socket.remoteAddress;
  return address === "127.0.0.1" || address === "::1" || address === "::ffff:127.0.0.1";
}

function localEntry(): Plugin {
  return {
    name: "local-entry",
    configureServer(server: ViteDevServer) {
      server.middlewares.use((request, response, next) => {
        const url = request.url?.split("?")[0];
        if (url !== "/__local/enter" || request.method !== "POST") return next();
        if (!loopback(request)) {
          response.statusCode = 403;
          response.end("Local entry is only available from this computer");
          return;
        }
        const file = localAdminFile();
        if (!file) {
          response.statusCode = 404;
          response.end("Local administrator file was not found");
          return;
        }
        void (async () => {
          const stored = JSON.parse(fs.readFileSync(file, "utf8")) as {
            username?: string;
            password?: string;
          };
          const upstream = await fetch("http://127.0.0.1:8081/api/v2/auth/login", {
            method: "POST",
            headers: {
              "content-type": "application/json",
              origin: "http://127.0.0.1:8081",
              host: "127.0.0.1:8081",
            },
            body: JSON.stringify({
              username: stored.username,
              password: stored.password,
            }),
          });
          response.statusCode = upstream.status;
          for (const cookie of upstream.headers.getSetCookie?.() ?? [])
            response.appendHeader("set-cookie", cookie);
          response.setHeader("content-type", "application/json; charset=utf-8");
          response.setHeader("cache-control", "no-store");
          response.end(
            upstream.ok
              ? JSON.stringify({ ok: true })
              : JSON.stringify({ ok: false }),
          );
        })().catch(() => {
          if (response.headersSent) return;
          response.statusCode = 502;
          response.end(JSON.stringify({ ok: false }));
        });
      });
    },
  };
}

export default defineConfig({
  plugins: [react(), localEntry()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8081",
        changeOrigin: true,
        configure(proxy) {
          proxy.on("proxyReq", (upstream, request) => {
            // Map only this loopback development origin to the local gateway.
            // Unrelated origins remain intact for the API to reject.
            const host = request.headers.host;
            if (host && /^(127\.0\.0\.1|localhost):\d+$/.test(host)
                && request.headers.origin === `http://${host}`) {
              upstream.setHeader("Origin", "http://127.0.0.1:8081");
            }
          });
        },
      },
    },
  },
  build: { chunkSizeWarningLimit: 450 },
});
