// app/scripts/generate-errors.js
import fs from "fs";
import path from "path";
import YAML from "yaml";
import { fileURLToPath } from "url";

// __dirname polyfill for ESM
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Paths
const errorsYamlPath = path.resolve(__dirname, "../shared/errors.yaml");
const outputTsPath = path.resolve(__dirname, "../src/lib/errorCodes.ts");
const outputPyPath = path.resolve(__dirname, "../../functions-python/game_logic/errors.py");

// Read YAML
const yamlContent = fs.readFileSync(errorsYamlPath, "utf8");
const errorDefs = YAML.parse(yamlContent);

// Extract codes
let errorCodes = [];
if (Array.isArray(errorDefs.codes)) {
  errorCodes = errorDefs.codes;
} else {
  errorCodes = Object.keys(errorDefs);
}

//
// --- TypeScript file ---
//
const tsLines = [
  "// Auto-generated from errors.yaml — do not edit manually",
  "export const ErrorCode = {",
  ...errorCodes.map(code => `  ${code}: "${code}",`),
  "} as const;",
  "",
  "export type ErrorCode = typeof ErrorCode[keyof typeof ErrorCode];",
  "",
];
fs.writeFileSync(outputTsPath, tsLines.join("\n"), "utf8");
console.log(`✅ Generated errorCodes.ts with ${errorCodes.length} codes`);

//
// --- Python file ---
//
const pyLines = [
  '# Auto-generated from errors.yaml — do not edit manually',
  'from enum import Enum',
  'from typing import Any, Dict, Optional',
  '',
  '',
  'class ErrorCode(str, Enum):',
  ...errorCodes.map(code => `    ${code} = "${code}"`),
  '',
  '',
  'class EngineError(Exception):',
  '    """Base exception for game engine errors.',
  '',
  '    Attributes:',
  '        code: ErrorCode enum',
  '        message: optional human message (not shown to client; for logs)',
  "        details: optional structured data (e.g. {'r':3,'c':2,'card':'AS'})",
  '    """',
  '',
  '    def __init__(',
  '        self,',
  '        code: ErrorCode,',
  '        message: Optional[str] = None,',
  '        details: Optional[Dict[str, Any]] = None',
  '    ):',
  '        self.code = code',
  '        self.details = details or {}',
  '        super().__init__(message or code.value)',
  '',
  '    def to_dict(self) -> Dict[str, Any]:',
  '        return {"code": self.code.value, "details": self.details}',
  '',
];
fs.writeFileSync(outputPyPath, pyLines.join("\n"), "utf8");
console.log(`✅ Generated errors.py with ${errorCodes.length} codes`);
