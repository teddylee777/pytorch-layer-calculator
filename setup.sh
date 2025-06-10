mkdir -p ~/.streamlit/
	
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
\n\
[theme]\n\
primaryColor = \"#ff6b6b\"\n\
backgroundColor = \"#FFFFFF\"\n\
secondaryBackgroundColor = \"#f0f2f6\"\n\
textColor = \"#262730\"\n\
font = \"sans serif\"\n\
base = \"light\"\n\
" > ~/.streamlit/config.toml