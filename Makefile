all: pip main
pip:
	@pip install -r requirements.txt > /dev/null
main:
	@python3 main.py
