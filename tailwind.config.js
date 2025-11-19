/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./templates/**/*.{html,js}",
    "./app.py",
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          light: "#c7d2fe",
          DEFAULT: "#4f46e5",
          dark: "#312e81",
        },
      },
    },
  },
  plugins: [],
}

