import { readFileSync } from "node:fs";
import { fileURLToPath, URL } from "node:url";

import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

const softwareVersion = readFileSync(fileURLToPath(new URL("../../VERSION", import.meta.url)), "utf8").trim();

export default defineConfig({
  root: fileURLToPath(new URL(".", import.meta.url)),
  plugins: [react()],
  define: {
    __CRABRAG_VERSION__: JSON.stringify(softwareVersion),
    __CRABRAG_VERSION_LABEL__: JSON.stringify(`v${softwareVersion}`),
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: "./src/test/setup.ts",
  },
});
