# Daily Expense Tracker

Flask application that records expenses, performs simple anomaly detection, and provides a 7-day Prophet forecast. Tailwind CSS now powers the UI.

## Getting started

1. **Python setup**
   - Create a virtual environment and install requirements listed in your preferred `requirements.txt` (not included here).
   - Run the Flask server:
     ```
     flask run
     ```

2. **Tailwind CSS**
   - Install Node dependencies (already defined in `package.json`):
     ```
     npm install
     ```
   - Build the CSS once:
     ```
     npm run build:css
     ```
   - Or keep Tailwind compiling as you edit templates:
     ```
     npm run watch:css
     ```

The compiled stylesheet lives in `static/css/tailwind.css` and is referenced by the Jinja templates.

## UI features

- Interactive dashboard with collapsible expense form, live search, and client-side sorting/filtering
- Category distribution bars and daily spend sparkline for at-a-glance trends
- Modernized forecast view with toggleable chart/table powered by Prophet output

