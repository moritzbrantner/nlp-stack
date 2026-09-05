import type { Config } from "tailwindcss";

const config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "../packages/nlp-app-ui/src/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        canvas: "var(--color-canvas)",
        surface: "var(--color-surface)",
        ink: "var(--color-ink)",
        muted: "var(--color-muted)",
        line: "var(--color-line)",
        accent: "var(--color-accent)",
        "accent-soft": "var(--color-accent-soft)",
      },
    },
  },
  plugins: [],
} satisfies Config;

export default config;
