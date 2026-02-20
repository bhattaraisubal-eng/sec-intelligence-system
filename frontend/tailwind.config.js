/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        'bb-black': '#0c0f14',
        'bb-panel': '#111520',
        'bb-surface': '#181d28',
        'bb-border': '#1e2433',
        'bb-hover': '#222838',
        'accent': '#14b8a6',
        'accent-dim': '#0d9488',
        'accent-bright': '#2dd4bf',
        'accent-bg': 'rgba(20, 184, 166, 0.08)',
        'amber': '#f59e0b',
        'amber-dim': '#d97706',
        'amber-bright': '#fbbf24',
        'amber-bg': 'rgba(245, 158, 11, 0.08)',
        'bb_blue': '#3b82f6',
        'bb_blue-bright': '#60a5fa',
        'bb_blue-dim': '#2563eb',
        'bb_blue-bg': 'rgba(59, 130, 246, 0.08)',
        'bb_red': '#ef4444',
        'bb_red-dim': '#dc2626',
        'bb_red-bg': 'rgba(239, 68, 68, 0.08)',
        'bb-gray': {
          100: '#f1f5f9',
          200: '#cbd5e1',
          300: '#94a3b8',
          400: '#64748b',
          500: '#475569',
          600: '#334155',
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
