run:
	bash scripts/run_all.sh

clean:
	rm -rf models reports data/processed && mkdir -p models reports/figures data/processed
