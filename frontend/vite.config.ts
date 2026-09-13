import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

const repositoryRoot = fileURLToPath(new URL("../", import.meta.url));

function readLocalCertificate(path: string) {
  if (!existsSync(path)) {
    throw new Error(`Local HTTPS file missing: ${path}. Follow the mkcert setup in the root README.`);
  }
  return readFileSync(path);
}

export default defineConfig(({ command, mode }) => {
  const env = loadEnv(mode, repositoryRoot, "");
  const productionApiBaseUrl = process.env.VITE_API_BASE_URL || env.VITE_API_BASE_URL;
  if (command === "build" && process.env.VERCEL && !productionApiBaseUrl) {
    throw new Error("Set VITE_API_BASE_URL to the production HTTPS backend origin on Vercel.");
  }
  if (
    command === "build" &&
    process.env.VERCEL &&
    productionApiBaseUrl &&
    !productionApiBaseUrl.startsWith("https://")
  ) {
    throw new Error("VITE_API_BASE_URL must use HTTPS on Vercel.");
  }
  const certificatePath = resolve(
    repositoryRoot,
    env.TLS_CERT_FILE || ".certs/localhost.pem",
  );
  const privateKeyPath = resolve(
    repositoryRoot,
    env.TLS_KEY_FILE || ".certs/localhost-key.pem",
  );
  const https =
    command === "serve"
      ? {
          cert: readLocalCertificate(certificatePath),
          key: readLocalCertificate(privateKeyPath),
        }
      : undefined;

  return {
    plugins: [react()],
    envDir: repositoryRoot,
    server: {
      host: "127.0.0.1",
      port: 5173,
      https,
      proxy: {
        "/api": {
          target: "https://127.0.0.1:8000",
          changeOrigin: true,
          // Both servers use the same local-only mkcert certificate.
          secure: false,
        },
      },
    },
  };
});
