import { resolve } from "node:path";

import type { NextConfig } from "next";

const isGitHubActions = process.env.GITHUB_ACTIONS === "true";
const basePath = isGitHubActions ? "/nlp-stack" : "";

const nextConfig: NextConfig = {
  output: "export",
  trailingSlash: true,
  images: {
    unoptimized: true,
  },
  basePath,
  env: {
    NEXT_PUBLIC_BASE_PATH: basePath,
  },
  turbopack: {
    root: resolve(process.cwd(), ".."),
  },
  ...(basePath ? { assetPrefix: `${basePath}/` } : {}),
};

export default nextConfig;
