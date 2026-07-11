import { existsSync } from "node:fs";
import { spawnSync } from "node:child_process";

const candidates = process.platform === "win32"
  ? [".venv/Scripts/python.exe", "python"]
  : [".venv/bin/python", "python3", "python"];
const executable = candidates.find((candidate) => candidate === "python" || candidate === "python3" || existsSync(candidate));

if (!executable) {
  console.error("No Python interpreter found. Create .venv or add Python to PATH.");
  process.exit(1);
}

const result = spawnSync(executable, ["-m", "pytest", "-q"], { stdio: "inherit" });
if (result.error) {
  console.error(result.error.message);
  process.exit(1);
}
process.exit(result.status ?? 1);
