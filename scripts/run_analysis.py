"""Run the fixed GAM family with 20 training bootstraps by default."""

from scripts.run_benchmark import main

if __name__ == "__main__":
    raise SystemExit(main(default_policy="fixed_gam"))
