import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // In dev, proxy /api to the Flask backend so the frontend code can just
    // call "/api/..." with no CORS fuss and no hardcoded localhost port.
    proxy: {
      "/api": "http://127.0.0.1:5000",
    },
  },
});
