/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        'bb-black': '#0a0a0a',
        'bb-panel': '#111111',
        'bb-surface': '#1a1a1a',
        'bb-border': '#1e1e1e',
        'bb-hover': '#222222',
        'term-green': '#00d26a',
        'term-dim': '#00a854',
        'term-bright': '#33ff99',
        'term-bg': 'rgba(0, 210, 106, 0.08)',
        'amber': '#ff9500',
        'amber-dim': '#cc7700',
        'amber-bright': '#ffb347',
        'amber-bg': 'rgba(255, 149, 0, 0.08)',
        'bb_blue': '#2962ff',
        'bb_blue-bright': '#448aff',
        'bb_blue-dim': '#1a44b8',
        'bb_blue-bg': 'rgba(41, 98, 255, 0.08)',
        'bb_red': '#ff3b30',
        'bb_red-dim': '#cc2f26',
        'bb_red-bg': 'rgba(255, 59, 48, 0.08)',
        'bb-gray': {
          100: '#f5f5f5',
          200: '#cccccc',
          300: '#999999',
          400: '#666666',
          500: '#444444',
          600: '#333333',
        },
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      fontSize: {
        'xxs': ['10px', { lineHeight: '14px' }],
      },
      keyframes: {
        'pulse-dot': {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.3' },
        },
        'scan-line': {
          '0%': { transform: 'translateY(-100%)' },
          '100%': { transform: 'translateY(100%)' },
        },
        blink: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0' },
        },
      },
      animation: {
        'pulse-dot': 'pulse-dot 1.5s ease-in-out infinite',
        'scan-line': 'scan-line 3s linear infinite',
        blink: 'blink 1s step-end infinite',
      },
    },
  },
  plugins: [],
};
