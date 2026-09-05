import type { Preview } from "@storybook/nextjs-vite";

import "../app/globals.css";

const preview: Preview = {
  parameters: {
    controls: {
      expanded: true,
    },
  },
};

export default preview;
