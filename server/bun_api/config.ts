import { resolve } from "node:path";

export interface GatewayConfig {
  ragBaseUrl: string;
  internalToken: string;
  subject: string;
  roles: string;
  groups: string;
  permissionRevision: string;
  localAdmin: boolean;
}

export interface GatewayEnvironment extends GatewayConfig {
  projectRoot: string;
  port: number;
}

export function readGatewayEnvironment(env: NodeJS.ProcessEnv = process.env): GatewayEnvironment {
  return {
    ragBaseUrl: env.RAG_BASE_URL ?? "http://127.0.0.1:8001",
    internalToken: env.CRABRAG_INTERNAL_TOKEN ?? "",
    subject: env.CRABRAG_SUBJECT ?? "local-user",
    roles: env.CRABRAG_ROLES ?? "",
    groups: env.CRABRAG_GROUPS ?? "",
    permissionRevision: env.CRABRAG_PERMISSION_REVISION ?? "1",
    localAdmin: (env.CRABRAG_LOCAL_ADMIN ?? "true").toLowerCase() !== "false",
    projectRoot: resolve(env.CRABRAG_ROOT ?? env.ELCQA_ROOT ?? process.cwd()),
    port: Number(env.PORT ?? 3003),
  };
}

export function ragHeaders(config: GatewayConfig, includeJson = false): Headers {
  const headers = new Headers();
  if (includeJson) headers.set("content-type", "application/json");
  headers.set("x-crabrag-internal-token", config.internalToken);
  headers.set("x-crabrag-subject", config.subject);
  headers.set("x-crabrag-roles", config.roles);
  headers.set("x-crabrag-groups", config.groups);
  headers.set("x-crabrag-permission-revision", config.permissionRevision);
  headers.set("x-crabrag-admin", String(config.localAdmin));
  return headers;
}
