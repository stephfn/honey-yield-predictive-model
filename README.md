# honey-yield-predictive-model
Predictive analytics pipeline for hive optimization using tree-based forecasting models.

## DuckDB Setup
To connect to our DuckDB server, you must install and configure Tailscale (VPN) and DuckDB.

**To install Tailscale:**
1. Contact @CmdrJorgs to get your Tailscale invite.
2. [Download the Tailscale installer](https://tailscale.com/download/) for your operating system.
3. Install. The installer will provide a link for you to sign into Tailscale. Login or create a new account to register your device.
4. Visit the URL provided by @CmdrJorgs to add your device to the team's DuckDB network.

**To install DuckDB (Python):**
1. In your python environment, run `pip install duckdb` (Default Anaconda installs already have DuckDB, in which case you may need to update it.)
2. Copy the `.env` file provided by @CmdrJorgs into your repo root directory.

The DuckDB scripts in the Jupyter notebook should work now.