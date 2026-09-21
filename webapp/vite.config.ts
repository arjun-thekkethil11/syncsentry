import react from '@vitejs/plugin-react'
// Import defineConfig from vitest/config, not plain vite, so the `test`
// key type-checks during `npm run build` (tsc -b && vite build).
import { defineConfig } from 'vitest/config'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    globals: true,
  },
})
