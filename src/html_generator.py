import os
import re

class HTMLGenerator:
    def __init__(self, output_dir, subject_name="Exam Paper Extraction", year="2021", paper_key="Paper 1"):
        """
        Premium HTML Generator that compiles extracted exam questions and paired mark schemes
        into a sleek, interactive workbook UI. This class is fully self-contained.
        """
        self.output_dir = os.path.abspath(output_dir)
        self.images_dir_name = "images"
        self.assets_dir_name = "_site_assets"
        
        self.subject_name = subject_name
        self.year = year
        self.paper_key = paper_key
        
        # Ensure directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, self.images_dir_name), exist_ok=True)
        self.write_static_assets()

    def write_static_assets(self):
        """
        Writes self-contained, custom premium CSS and JS files directly to the output folder.
        """
        target_assets_dir = os.path.join(self.output_dir, self.assets_dir_name)
        os.makedirs(target_assets_dir, exist_ok=True)

        # Custom Premium Stylesheet (site.css)
        css_content = """/* Premium Exam Workbook Design System */
:root {
  --font-family-title: 'Outfit', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  --font-family-body: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  
  /* Premium Warm Amber & Cream Color Scheme */
  --brand-indigo: hsl(38, 90%, 43%);
  --brand-indigo-hover: hsl(38, 90%, 35%);
  --brand-indigo-light: hsl(38, 70%, 96%);
  --accent-amber: hsl(38, 100%, 50%);
  --accent-color: var(--accent-amber);
  
  --bg-app: #fbfaf7;
  --bg-card: #ffffff;
  --bg-paper: #faf8f5;
  
  --text-dark: hsl(24, 25%, 15%);
  --text-body: hsl(24, 15%, 28%);
  --text-muted: hsl(24, 10%, 48%);
  
  --border-light: hsl(38, 15%, 88%);
  --border-focus: hsl(38, 90%, 82%);
  --border-color: var(--border-light);
  --sidebar-text-muted: var(--text-muted);
  
  /* Borders and Shadows */
  --shadow-sm: 0 1px 3px rgba(0,0,0,0.02), 0 1px 2px rgba(0,0,0,0.04);
  --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.04), 0 2px 4px -1px rgba(0,0,0,0.02);
  --shadow-lg: 0 10px 25px -5px rgba(0,0,0,0.05), 0 8px 16px -8px rgba(0,0,0,0.03);
  --shadow-hover: 0 20px 25px -5px rgba(243, 75, 59, 0.06), 0 10px 10px -5px rgba(0,0,0,0.02);
  
  --radius-sm: 6px;
  --radius-md: 10px;
  --radius-lg: 16px;
  
  --danger-color: hsl(0, 84%, 60%);
  --success-color: hsl(150, 100%, 30%);
}

* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

body {
  font-family: var(--font-family-body);
  background-color: var(--bg-app);
  color: var(--text-body);
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

h1, h2, h3, h4, h5, h6 {
  font-family: var(--font-family-title);
  color: var(--text-dark);
}

/* Custom Scrollbars */
::-webkit-scrollbar {
  width: 6px;
  height: 6px;
}
::-webkit-scrollbar-track {
  background: transparent;
}
::-webkit-scrollbar-thumb {
  background: rgba(0, 0, 0, 0.08);
  border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover {
  background: rgba(0, 0, 0, 0.16);
}

/* Dashboard Layout */
.dashboard-container {
  max-width: 1100px;
  margin: 3rem auto;
  padding: 0 1.5rem;
}

.dashboard-header {
  background: linear-gradient(135deg, hsl(222, 47%, 11%) 0%, hsl(222, 47%, 18%) 100%);
  color: white;
  padding: 3.5rem 3rem;
  border-radius: var(--radius-lg);
  margin-bottom: 2.5rem;
  box-shadow: var(--shadow-lg);
  border-bottom: 5px solid var(--brand-indigo);
  position: relative;
  overflow: hidden;
}

.dashboard-header h1 {
  font-size: 2.5rem;
  font-weight: 800;
  color: white;
  margin-bottom: 0.75rem;
  letter-spacing: -0.75px;
}

.dashboard-header p {
  color: var(--sidebar-text-muted);
  font-size: 1.1rem;
  font-weight: 400;
  opacity: 0.85;
}

.eyebrow {
  color: var(--accent-amber);
  font-family: var(--font-family-title);
  font-size: 0.85rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 1.5px;
  margin-bottom: 0.5rem;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 1.5rem;
  margin-bottom: 3rem;
}

.stat-card {
  background: var(--bg-card);
  border: 1px solid var(--border-light);
  border-radius: var(--radius-md);
  padding: 1.75rem;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  box-shadow: var(--shadow-sm);
  transition: transform 0.25s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.25s ease, border-color 0.25s ease;
}

.stat-card:hover {
  transform: translateY(-4px);
  box-shadow: var(--shadow-hover);
  border-color: var(--brand-indigo);
}

.stat-card h3 {
  font-size: 2.5rem;
  font-weight: 800;
  color: var(--text-dark);
  margin-bottom: 0.25rem;
  letter-spacing: -1px;
}

.stat-card p {
  font-size: 0.75rem;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 1px;
  font-weight: 700;
}

.search-toolbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 2rem;
  flex-wrap: wrap;
  gap: 1rem;
  border-bottom: 2px solid var(--border-light);
  padding-bottom: 1rem;
}

.search-toolbar h3 {
  font-size: 1.5rem;
  font-weight: 700;
  letter-spacing: -0.5px;
}

.search-input {
  padding: 0.7rem 1.25rem;
  border: 1.5px solid var(--border-light);
  border-radius: var(--radius-md);
  font-size: 0.95rem;
  font-family: inherit;
  width: 100%;
  max-width: 380px;
  box-shadow: var(--shadow-sm);
  transition: border-color 0.2s, box-shadow 0.2s;
}

.search-input:focus {
  outline: none;
  border-color: var(--brand-indigo);
  box-shadow: 0 0 0 4px var(--border-focus);
}

.questions-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 1.5rem;
}

.question-card {
  background: var(--bg-card);
  border: 1.5px solid var(--border-light);
  border-radius: var(--radius-md);
  padding: 2rem;
  box-shadow: var(--shadow-sm);
  display: flex;
  flex-direction: column;
  transition: transform 0.25s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.25s ease, border-color 0.25s ease;
}

.question-card:hover {
  transform: translateY(-4px);
  box-shadow: var(--shadow-hover);
  border-color: var(--brand-indigo);
}

.card-meta {
  display: flex;
  gap: 0.75rem;
  font-size: 0.85rem;
  color: var(--text-muted);
  margin-bottom: 0.75rem;
  align-items: center;
  flex-wrap: wrap;
  font-weight: 500;
}

.card-badge-container {
  display: flex;
  gap: 0.4rem;
  margin-top: 0.25rem;
  margin-left: auto;
}

.card-badge {
  padding: 0.15rem 0.5rem;
  border-radius: var(--radius-sm);
  font-size: 0.72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.badge-marks { background-color: hsl(0, 100%, 97%); color: hsl(0, 84%, 60%); border: 1.5px solid hsl(0, 100%, 90%); }
.badge-mcq { background-color: hsl(210, 100%, 96%); color: hsl(210, 100%, 45%); border: 1.5px solid hsl(210, 100%, 90%); }
.badge-table { background-color: hsl(150, 100%, 96%); color: hsl(150, 100%, 30%); border: 1.5px solid hsl(150, 100%, 90%); }
.badge-diagram { background-color: hsl(45, 100%, 96%); color: hsl(35, 100%, 40%); border: 1.5px solid hsl(45, 100%, 88%); }

.question-card h2 {
  font-size: 1.4rem;
  color: var(--text-dark);
  margin-bottom: 0.75rem;
  font-weight: 700;
  letter-spacing: -0.25px;
}

.question-card h2 a {
  color: var(--text-dark);
  text-decoration: none;
  transition: color 0.15s ease;
}

.question-card h2 a:hover {
  color: var(--brand-indigo);
}

.question-card p {
  color: var(--text-body);
  font-size: 0.98rem;
  margin-bottom: 1.5rem;
  line-height: 1.6;
  flex-grow: 1;
}

.btn-link {
  display: inline-block;
  background-color: var(--brand-indigo);
  color: white;
  padding: 0.55rem 1.35rem;
  border-radius: var(--radius-sm);
  text-decoration: none;
  font-weight: 700;
  font-size: 0.85rem;
  font-family: var(--font-family-title);
  letter-spacing: 0.5px;
  box-shadow: 0 2px 4px rgba(2, 132, 199, 0.1);
  transition: background-color 0.2s, transform 0.2s, box-shadow 0.2s;
  text-align: center;
}

.btn-link:hover {
  background-color: var(--brand-indigo-hover);
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(243, 75, 59, 0.2);
}

/* 3-Column Layout */
.app-container {
  display: flex;
  flex-direction: column;
  height: 100vh;
  overflow: hidden;
}

.app-header {
  background-color: var(--bg-card);
  color: var(--text-dark);
  padding: 0 2rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 2px solid var(--border-light);
  flex-shrink: 0;
  height: 60px;
  box-shadow: 0 1px 2px rgba(0,0,0,0.02);
}

.app-header h1 {
  font-size: 1.15rem;
  font-weight: 700;
  color: var(--text-dark);
  letter-spacing: -0.25px;
}

.app-header-meta {
  font-size: 0.85rem;
  color: var(--text-muted);
  display: flex;
  gap: 0.75rem;
  align-items: center;
}

.app-header-meta a {
  background-color: var(--bg-app);
  color: var(--text-body);
  text-decoration: none;
  font-weight: 600;
  padding: 0.35rem 0.85rem;
  border-radius: var(--radius-sm);
  transition: background-color 0.2s, color 0.2s;
  font-family: var(--font-family-title);
  border: 1px solid var(--border-light);
}

.app-header-meta a:hover {
  background-color: var(--brand-indigo);
  color: white;
  border-color: var(--brand-indigo);
}

.layout-grid {
  display: flex;
  flex: 1;
  overflow: hidden;
}

/* Sidebar Styling */
.sidebar {
  width: 250px;
  background-color: var(--bg-paper);
  color: var(--text-body);
  display: flex;
  flex-direction: column;
  border-right: 1px solid var(--border-light);
  flex-shrink: 0;
}

.sidebar-search-container {
  padding: 1rem;
  border-bottom: 1.5px solid var(--border-light);
}

.sidebar-search {
  width: 100%;
  padding: 0.55rem 0.85rem;
  background-color: var(--bg-card);
  border: 1.5px solid var(--border-light);
  border-radius: var(--radius-sm);
  color: var(--text-dark);
  font-size: 0.85rem;
  font-family: inherit;
  transition: border-color 0.2s;
}

.sidebar-search:focus {
  outline: none;
  border-color: var(--brand-indigo);
}

.sidebar-list {
  flex: 1;
  overflow-y: auto;
  padding: 0.75rem;
}

.sidebar-list::-webkit-scrollbar-thumb {
  background: rgba(0, 0, 0, 0.08);
}
.sidebar-list::-webkit-scrollbar-thumb:hover {
  background: rgba(0, 0, 0, 0.16);
}

.sidebar-item {
  display: block;
  padding: 0.55rem 0.85rem;
  color: var(--text-muted);
  text-decoration: none;
  border-radius: var(--radius-sm);
  font-size: 0.88rem;
  font-weight: 500;
  margin-bottom: 0.25rem;
  transition: background-color 0.15s, color 0.15s;
}

.sidebar-item:hover {
  background-color: var(--brand-indigo-light);
  color: var(--brand-indigo);
}

.sidebar-item.active {
  background-color: var(--brand-indigo);
  color: white;
  font-weight: 600;
}

.sidebar-meta {
  padding: 1rem;
  background-color: var(--bg-app);
  border-top: 1.5px solid var(--border-light);
  font-size: 0.75rem;
  color: var(--text-muted);
}

.sidebar-meta-row {
  margin-bottom: 0.4rem;
  display: flex;
  justify-content: space-between;
}

.sidebar-meta-row span:first-child {
  font-weight: 700;
  text-transform: uppercase;
  font-size: 0.68rem;
  letter-spacing: 0.5px;
}

/* Question and Answer Columns */
.column-pane {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  background-color: white;
}

.column-pane-header {
  padding: 0 1.5rem;
  background-color: #f8fafc;
  border-bottom: 1.5px solid var(--border-light);
  font-weight: 700;
  color: var(--text-dark);
  font-size: 1rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-shrink: 0;
  height: 52px;
  font-family: var(--font-family-title);
  letter-spacing: -0.2px;
}

.column-pane-content {
  flex: 1;
  overflow-y: auto;
  padding: 2.5rem;
}

/* Answer Column */
.answer-pane {
  border-left: 2px solid var(--border-light);
  background-color: var(--bg-paper);
}

.toggle-answer-btn {
  background-color: var(--brand-indigo);
  color: white;
  border: none;
  padding: 0.4rem 1rem;
  border-radius: var(--radius-sm);
  cursor: pointer;
  font-weight: 700;
  font-size: 0.82rem;
  font-family: var(--font-family-title);
  letter-spacing: 0.5px;
  transition: background-color 0.2s, transform 0.15s;
}

.toggle-answer-btn:hover {
  background-color: var(--brand-indigo-hover);
}

/* Real Question Paper Styling */
.question-part-parent {
  font-size: 1.18rem;
  font-weight: 700;
  color: var(--text-dark);
  line-height: 1.75;
  margin-bottom: 2rem;
  border-left: 4.5px solid var(--brand-indigo);
  padding: 1.25rem 1.5rem;
  background: linear-gradient(to right, var(--brand-indigo-light), transparent);
  border-radius: 4px;
  box-shadow: inset 1px 0 0 rgba(0,0,0,0.03);
}

.question-part {
  display: flex;
  background-color: var(--bg-card);
  border: 1px solid var(--border-light);
  border-radius: var(--radius-md);
  padding: 1.5rem 1.75rem;
  margin-bottom: 1.5rem;
  box-shadow: var(--shadow-sm);
  transition: transform 0.2s, box-shadow 0.2s;
}

.question-part:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(35, 30, 20, 0.04);
}

.part-label {
  min-width: 32px;
  height: 32px;
  width: auto;
  padding: 0 8px;
  background-color: var(--brand-indigo-light);
  color: var(--brand-indigo);
  border-radius: 16px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-weight: 800;
  font-size: 0.92rem;
  font-family: var(--font-family-title);
  flex-shrink: 0;
  border: 1px solid var(--border-focus);
  margin-right: 1.25rem;
  box-shadow: 0 2px 4px rgba(220, 130, 10, 0.04);
}

.part-body {
  flex: 1;
  font-size: 1.08rem;
  line-height: 1.7;
  color: var(--text-body);
}

.part-marks {
  display: inline-block;
  background-color: var(--brand-indigo-light);
  color: var(--brand-indigo);
  font-weight: 700;
  font-size: 0.76rem;
  padding: 0.2rem 0.6rem;
  border-radius: 20px;
  border: 1px solid var(--border-focus);
  margin-left: 0.5rem;
  vertical-align: middle;
  letter-spacing: 0.2px;
}

.part-body p {
  margin-bottom: 1rem;
}

.part-marks {
  font-weight: 700;
  color: var(--danger-color);
  font-size: 0.95rem;
  margin-left: 0.5rem;
  font-family: var(--font-family-title);
}

/* Intro blocks styling (for sub-question intros like 1(b) or 2(c)) */
.question-part-intro {
  display: flex;
  background-color: transparent;
  border: none;
  padding: 0.5rem 1.75rem;
  margin-top: 1.5rem;
  margin-bottom: 0.5rem;
}

.part-label-intro {
  min-width: 32px;
  height: 32px;
  width: auto;
  padding: 0 8px;
  background-color: var(--brand-indigo-light);
  color: var(--brand-indigo);
  border-radius: 16px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-weight: 800;
  font-size: 0.92rem;
  font-family: var(--font-family-title);
  flex-shrink: 0;
  border: 1px solid var(--border-focus);
  margin-right: 1.25rem;
  box-shadow: 0 2px 4px rgba(220, 130, 10, 0.04);
}

.part-body-intro {
  flex: 1;
  font-size: 1.08rem;
  line-height: 1.7;
  color: var(--text-dark);
  font-weight: 600;
}

.part-body-intro p {
  margin-bottom: 1rem;
}

/* Hierarchical indentation for sub-question nested elements */
.question-part.nested {
  margin-left: 2.5rem;
}

.question-part-intro.nested {
  margin-left: 2.5rem;
}

@media (max-width: 768px) {
  .question-part.nested {
    margin-left: 1.25rem;
  }
  .question-part-intro.nested {
    margin-left: 1.25rem;
  }
}

.figure-wrapper {
  margin: 1.5rem 0;
  background-color: white;
  border: 1.5px solid var(--border-light);
  padding: 1.25rem;
  border-radius: var(--radius-md);
  box-shadow: var(--shadow-sm);
  text-align: center;
  transition: border-color 0.2s;
}

.figure-wrapper:hover {
  border-color: var(--brand-indigo);
}

.figure-wrapper img {
  max-width: 100%;
  height: auto;
  border-radius: 4px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.03);
}

.figure-caption {
  font-size: 0.85rem;
  color: var(--text-muted);
  margin-top: 0.5rem;
  font-style: italic;
  font-weight: 500;
}

/* Premium styled Data Tables */
.table-wrapper {
  overflow-x: auto;
  margin: 1.5rem 0;
  border: 1.5px solid var(--border-light);
  border-radius: var(--radius-md);
  box-shadow: var(--shadow-sm);
}

.table-wrapper table {
  width: 100%;
  border-collapse: collapse;
  text-align: left;
  background-color: white;
  font-size: 0.95rem;
}

.table-wrapper th, .table-wrapper td {
  padding: 0.85rem 1.1rem;
  border-bottom: 1px solid var(--border-light);
}

.table-wrapper th {
  background-color: var(--brand-indigo-light);
  color: var(--text-dark);
  font-weight: 700;
  font-family: var(--font-family-title);
  letter-spacing: -0.2px;
}

.table-wrapper tr:hover td {
  background-color: #fafbfc;
}

.table-wrapper tr:last-child td {
  border-bottom: none;
}

/* MCQ Options styling */
.mcq-options-container {
  list-style: none;
  margin: 1.5rem 0;
  display: flex;
  flex-direction: column;
  gap: 0.85rem;
}

.mcq-option-item {
  display: flex;
  align-items: center;
  background-color: var(--bg-paper);
  border: 1px solid var(--border-light);
  border-radius: var(--radius-sm);
  padding: 0.85rem 1.25rem;
  transition: background-color 0.2s, border-color 0.2s, transform 0.15s;
  cursor: pointer;
}

.mcq-option-item:hover {
  background-color: var(--brand-indigo-light);
  border-color: var(--border-focus);
  transform: translateX(3px);
}

.mcq-radio-input {
  margin-right: 1rem;
  cursor: pointer;
  width: 1.25rem;
  height: 1.25rem;
  accent-color: var(--brand-indigo);
}

.mcq-option-label {
  cursor: pointer;
  font-size: 1.02rem;
  display: inline-flex;
  gap: 0.65rem;
  font-weight: 500;
  color: var(--text-body);
  flex: 1;
  user-select: none;
}

.option-letter {
  font-weight: 800;
  color: var(--brand-indigo);
}

/* Mark Scheme styling */
.answer-section {
  background: #fffdfb;
  border: 1px solid var(--border-light);
  border-left: 4.5px solid var(--brand-indigo);
  border-radius: var(--radius-md);
  padding: 1.5rem;
  margin-bottom: 1.5rem;
  box-shadow: var(--shadow-sm);
  transition: border-color 0.2s, box-shadow 0.2s;
}

.answer-section:hover {
  border-color: var(--brand-indigo);
  box-shadow: 0 4px 12px rgba(220, 130, 10, 0.06);
}

.answer-part-header {
  font-size: 0.88rem;
  font-weight: 800;
  color: var(--brand-indigo);
  text-transform: uppercase;
  letter-spacing: 0.8px;
  border-bottom: 1.5px solid var(--border-light);
  padding-bottom: 0.45rem;
  margin-bottom: 0.95rem;
  font-family: var(--font-family-title);
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.answer-part-header::before {
  content: "MS";
  background-color: var(--brand-indigo-light);
  padding: 0.15rem 0.4rem;
  border-radius: 4px;
  font-size: 0.72rem;
  font-weight: 900;
  letter-spacing: 0.5px;
}

.answer-part-body {
  font-size: 1rem;
  line-height: 1.65;
  color: var(--text-body);
}

.ms-para {
  font-size: 1.02rem;
  line-height: 1.65;
  color: var(--text-dark);
  margin-bottom: 0.85rem;
}

.ms-list {
  margin-left: 1.25rem;
  margin-bottom: 1.25rem;
  list-style-type: square;
}

.ms-list li {
  margin-bottom: 0.45rem;
  font-size: 1.02rem;
  color: var(--text-dark);
}

.no-results-sidebar {
  display: none;
  color: var(--sidebar-text-muted);
  text-align: center;
  padding: 1.5rem 1rem;
  font-size: 0.85rem;
}

/* Tablet and Mobile Breakpoints Stacking */
@media (max-width: 992px) {
  .layout-grid {
    flex-direction: column;
    overflow-y: auto;
  }
  
  .sidebar {
    width: 100%;
    height: auto;
    max-height: 250px;
    border-right: none;
    border-bottom: 1.5px solid hsl(222, 47%, 16%);
  }
  
  .column-pane {
    height: auto;
    overflow: visible;
  }
  
  .column-pane-content {
    overflow-y: visible;
    height: auto;
  }
  
  .answer-pane {
    border-left: none;
    border-top: 2px solid var(--border-light);
  }
}
"""

        # 2. Custom Interactive Scripts (site.js)
        js_content = """document.addEventListener('DOMContentLoaded', function() {
  // Toggle answer functionality
  const toggleBtn = document.getElementById('toggleAnswerBtn');
  const answerContent = document.getElementById('answerContent');
  
  if (toggleBtn && answerContent) {
    toggleBtn.addEventListener('click', function() {
      const isHidden = answerContent.style.display === 'none';
      if (isHidden) {
        answerContent.style.display = '';
        toggleBtn.textContent = 'Hide Answer';
      } else {
        answerContent.style.display = 'none';
        toggleBtn.textContent = 'Reveal Answer';
      }
    });
  }

  // Sidebar Question search query filtering
  const sidebarSearch = document.getElementById('sidebarSearch');
  const sidebarContainer = document.getElementById('sidebarContainer');
  const noResultsText = document.getElementById('noResultsSidebar');
  
  if (sidebarSearch && sidebarContainer) {
    sidebarSearch.addEventListener('input', function() {
      const query = this.value.toLowerCase().trim();
      const items = sidebarContainer.querySelectorAll('.sidebar-item');
      let visibleCount = 0;
      
      items.forEach(function(item) {
        const text = item.textContent.toLowerCase();
        if (!query || text.includes(query)) {
          item.style.display = '';
          visibleCount++;
        } else {
          item.style.display = 'none';
        }
      });
      
      if (noResultsText) {
        noResultsText.style.display = (visibleCount === 0) ? 'block' : 'none';
      }
    });
  }

  // Keyboard shortcut: Press 'A' to toggle Answer
  document.addEventListener('keydown', function(e) {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    
    if (e.key.toLowerCase() === 'a') {
      const toggleBtn = document.getElementById('toggleAnswerBtn');
      if (toggleBtn) {
        e.preventDefault();
        toggleBtn.click();
      }
    }
  });
});
"""

        with open(os.path.join(target_assets_dir, "site.css"), "w", encoding="utf-8") as f:
            f.write(css_content)
        with open(os.path.join(target_assets_dir, "site.js"), "w", encoding="utf-8") as f:
            f.write(js_content)
        print("Workbook generator wrote CSS/JS static assets successfully.")

    def _get_q_filename(self, q_num):
        """
        Standardizes safe question file links.
        e.g. "2(a)(ii)" -> "q_2aii.html"
        """
        clean = re.sub(r'[^a-zA-Z0-9]', '', str(q_num)).lower()
        return f"q_{clean}.html"

    def generate_qna_page(self, qna, all_qnas, current_idx):
        """
        Generates a premium 3-column QnA page matching the workbook CSS layout rules.
        """
        q_num = qna["question_number"]
        clean_filename = self._get_q_filename(q_num)
        
        # Build Navigation items
        nav_question_items = []
        for idx, item in enumerate(all_qnas):
            item_num = item["question_number"]
            item_filename = self._get_q_filename(item_num)
            is_current = (idx == current_idx)
            
            css_class = "sidebar-item active" if is_current else "sidebar-item"
            nav_question_items.append(
                f'<a class="{css_class}" href="{item_filename}">Question {item_num}</a>'
            )
            
        nav_question_list_html = "\n".join(nav_question_items)

        # Diagram elements (handled inline inside text_html)
        images_html = ""

        # Marks badge
        marks_badge = f'<span class="card-badge badge-marks">{qna["marks"]} Marks</span>' if qna.get("marks") else ""

        # Left Navigation Sidebar Metadata Card
        metadata_html = f"""
        <div class="sidebar-meta">
          <div class="sidebar-meta-row">
            <span>Subject</span>
            <span>{self.subject_name[:20]}</span>
          </div>
          <div class="sidebar-meta-row">
            <span>Year</span>
            <span>{self.year}</span>
          </div>
          <div class="sidebar-meta-row">
            <span>Paper</span>
            <span>{self.paper_key}</span>
          </div>
        </div>"""

        # Compile final layout
        html_content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Question {q_num} | {self.subject_name}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="{self.assets_dir_name}/site.css">
  
  <!-- MathJax Configuration for Chemistry/Physics Formulas -->
  <script>
    window.MathJax = {{
      tex: {{
        inlineMath: [['\\\\(', '\\\\)'], ['$', '$']],
        displayMath: [['\\\\[', '\\\\]'], ['$$', '$$']]
      }}
    }};
  </script>
  <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
</head>
<body>
  <div class="app-container">
    <!-- Top banner header -->
    <header class="app-header">
      <h1>{self.subject_name}</h1>
      <div class="app-header-meta">
        <span>{self.year} &bull; {self.paper_key}</span>
        <span>&bull;</span>
        <a href="index.html">Back to Dashboard</a>
      </div>
    </header>

    <!-- 3-Column layout -->
    <div class="layout-grid">
      <!-- COLUMN 1: LEFT NAVIGATION SIDEBAR -->
      <aside class="sidebar">
        <div class="sidebar-search-container">
          <input type="search" id="sidebarSearch" class="sidebar-search" placeholder="Filter questions...">
        </div>
        <div class="sidebar-list" id="sidebarContainer">
          {nav_question_list_html}
          <div class="no-results-sidebar" id="noResultsSidebar">No matches found</div>
        </div>
        {metadata_html}
      </aside>

      <!-- COLUMN 2: CENTER QUESTION PANEL -->
      <section class="column-pane">
        <div class="column-pane-header">
          <span>Question {q_num}</span>
          {marks_badge}
        </div>
        <div class="column-pane-content">
          <div class="markdown-body">
            {qna["text_html"]}
            {images_html}
          </div>
        </div>
      </section>

      <!-- COLUMN 3: RIGHT COLLAPSED ANSWER PANEL -->
      <section class="column-pane answer-pane">
        <div class="column-pane-header">
          <span>Mark Scheme Guideline</span>
          <button id="toggleAnswerBtn" class="toggle-answer-btn" type="button">Reveal Answer</button>
        </div>
        <div class="column-pane-content" id="answerContent" style="display: none;">
          <div class="markdown-body">
            {qna["answer_html"]}
          </div>
        </div>
      </section>
    </div>
  </div>
  <script src="{self.assets_dir_name}/site.js"></script>
</body>
</html>
"""
        filepath = os.path.join(self.output_dir, clean_filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)
        return clean_filename

    def generate_dashboard(self, qnas):
        """
        Generates index.html dashboard listing statistics and filtered searchable questions.
        """
        total_questions = len(qnas)
        mcq_count = sum(1 for qna in qnas if qna.get("is_mcq"))
        diagram_count = sum(1 for qna in qnas if qna.get("associated_images"))
        table_count = sum(1 for qna in qnas if "table_wrapper" in qna["text_html"])

        question_cards = ""
        for idx, qna in enumerate(qnas):
            q_num = qna["question_number"]
            clean_filename = self._get_q_filename(q_num)
            
            # Formulate tag badges
            badges = []
            if qna.get("marks"):
                badges.append(f'<span class="card-badge badge-marks">{qna["marks"]} Marks</span>')
            if qna.get("is_mcq"):
                badges.append('<span class="card-badge badge-mcq">MCQ</span>')
            if "table_wrapper" in qna["text_html"]:
                badges.append('<span class="card-badge badge-table">Table</span>')
            if qna.get("associated_images"):
                badges.append('<span class="card-badge badge-diagram">Diagram</span>')
            
            badges_html = " ".join(badges)
            
            # Strip tags for a clean text preview
            plain_text = re.sub(r'<[^>]*>', ' ', qna["text_html"])
            plain_text = re.sub(r'\s+', ' ', plain_text).strip()
            snippet = plain_text[:160] + "..." if len(plain_text) > 160 else plain_text
            search_str = f"question {q_num} {plain_text}".lower()

            question_cards += f"""
            <article class="question-card" data-search-item data-search="{search_str}">
              <div class="card-meta">
                <span>{self.year}</span>
                <span>&bull;</span>
                <span>{self.paper_key}</span>
                <span>&bull;</span>
                <span>Question {q_num}</span>
                <div class="card-badge-container">
                  {badges_html}
                </div>
              </div>
              <h2><a href="{clean_filename}">{self.subject_name} - Question {q_num}</a></h2>
              <p>{snippet}</p>
              <a href="{clean_filename}" class="btn-link">Solve Question</a>
            </article>
            """

        html_content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{self.subject_name} | Interactive Workbook</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="{self.assets_dir_name}/site.css">
  <style>
    .dashboard-stat-card {{
      background: white;
      border: 1px solid var(--border-color);
      border-radius: var(--radius-md);
      padding: 1.5rem;
      text-align: center;
      transition: transform 0.2s, box-shadow 0.2s;
      box-shadow: var(--shadow-sm);
    }}
    .dashboard-stat-card:hover {{
      transform: translateY(-2px);
      box-shadow: var(--shadow-md);
      border-color: var(--brand-indigo);
    }}
  </style>
</head>
<body>
  <div class="dashboard-container">
    <!-- Header Hero banner -->
    <header class="dashboard-header">
      <p class="eyebrow" style="color: var(--accent-amber); font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem;">Interactive Exam Workbook</p>
      <h1>{self.subject_name}</h1>
      <p style="margin: 0;">Static workbook index for exam review and Q&A side-by-side study.</p>
    </header>

    <!-- Key Statistics Row -->
    <section class="stats-grid">
      <div class="dashboard-stat-card">
        <p class="eyebrow" style="color: var(--text-muted); font-size: 0.75rem; font-weight: 700; margin-bottom: 0.5rem;">Total Questions</p>
        <h2 style="font-size: 2.2rem; color: var(--text-dark); font-weight: 800;">{total_questions}</h2>
      </div>
      <div class="dashboard-stat-card">
        <p class="eyebrow" style="color: var(--text-muted); font-size: 0.75rem; font-weight: 700; margin-bottom: 0.5rem;">Multiple Choice</p>
        <h2 style="font-size: 2.2rem; color: var(--brand-indigo); font-weight: 800;">{mcq_count}</h2>
      </div>
      <div class="dashboard-stat-card">
        <p class="eyebrow" style="color: var(--text-muted); font-size: 0.75rem; font-weight: 700; margin-bottom: 0.5rem;">Diagram Exercises</p>
        <h2 style="font-size: 2.2rem; color: var(--accent-color); font-weight: 800;">{diagram_count}</h2>
      </div>
      <div class="dashboard-stat-card">
        <p class="eyebrow" style="color: var(--text-muted); font-size: 0.75rem; font-weight: 700; margin-bottom: 0.5rem;">Data Tables</p>
        <h2 style="font-size: 2.2rem; color: var(--success-color); font-weight: 800;">{table_count}</h2>
      </div>
    </section>

    <!-- Question List Section -->
    <section>
      <div class="search-toolbar">
        <h3>Question Sheets</h3>
        <input class="search-input" id="dashboardSearch" type="search" placeholder="Filter questions by keywords...">
      </div>

      <div class="questions-grid" id="questionsContainer" data-search-container>
        {question_cards}
        <div class="no-results" id="noResultsText" style="display: none; padding: 3rem; text-align: center; color: var(--text-muted); border: 2px dashed var(--border-color); border-radius: var(--radius-md); background: white; font-size: 1.05rem; width: 100%;">
          No questions match your current search query.
        </div>
      </div>
    </section>
  </div>

  <script>
    document.addEventListener('DOMContentLoaded', function() {{
      const searchInput = document.getElementById('dashboardSearch');
      const cards = document.querySelectorAll('.question-card');
      const noResults = document.getElementById('noResultsText');

      if (searchInput) {{
        searchInput.addEventListener('input', function() {{
          const query = this.value.toLowerCase().trim();
          let visibleCount = 0;

          cards.forEach(card => {{
            const searchContent = card.getAttribute('data-search') || '';
            if (!query || searchContent.includes(query)) {{
              card.style.display = '';
              visibleCount++;
            }} else {{
              card.style.display = 'none';
            }}
          }});

          if (visibleCount === 0) {{
            noResults.style.display = 'block';
          }} else {{
            noResults.style.display = 'none';
          }}
        }});
      }}
    }});
  </script>
</body>
</html>
"""
        filepath = os.path.join(self.output_dir, "index.html")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"Generated workbook index dashboard at: {filepath}")
        return filepath
