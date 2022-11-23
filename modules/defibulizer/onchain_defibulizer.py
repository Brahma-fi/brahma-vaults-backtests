"""
A module for aggregating the data from on-chain sources.
#####################################################################
Data format

"""
# pylint: disable=invalid-name, no-else-return, consider-using-with, unspecified-encoding, invalid-sequence-index, too-many-locals, unused-argument

import json
import os
import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery
from google.oauth2 import service_account
from web3 import Web3
from web3.middleware import geth_poa_middleware

load_dotenv()

class UniswapV3Data:  # pragma: no cover
    """
    Class for the extraction of mainnet data from the blockchain-etl BigQuery datasets.
    """

    def __init__(self, path, project_id, key_path) -> None:
        """
        Class constructor.
        """
        credentials = service_account.Credentials.from_service_account_file(key_path)
        self.client = bigquery.Client(credentials=credentials, project=project_id)
        self.default_dir = path

    def get_swap_data(self, address, download_latest=False):
        """
        Loads any locally stored datafiles and downloads lastest data if required.

        :param address: (str) Contract address of uniswap pool.
        :param download_latest: (bool) Whether to download latest data or just use local data.

        :return: (DataFrame) DataFrame of all swap event data
        """

        # check if file already exists
        files = os.listdir(self.default_dir)
        end_block = 0
        current_block = 0
        for file in files:
            if address in file:
                _, _, current_block = file.split('-')
                current_block = int(current_block[:-4])
                end_block = max(current_block, end_block)

        if download_latest:
            # if yes download from the last snap shot
            swap_query = f"""
                select * from `blockchain-etl.ethereum_uniswap.UniswapV3Pool_event_Swap`
                where contract_address='{address}' and block_number > {end_block}
                order by block_number;
                """ if end_block else f"""
                select * from `blockchain-etl.ethereum_uniswap.UniswapV3Pool_event_Swap`
                where contract_address='{address}'
                order by block_number;
                """
            query_job = self.client.query(swap_query)
            output = query_job.result().to_dataframe()
            output.to_csv(
                f'{self.default_dir}/{address}-{output.block_number.iloc[0]}-{output.block_number.iloc[-1]}.csv',
                index=False)

        # check if directory again exists
        files = os.listdir(self.default_dir)
        swap_data_list = []
        for file in files:
            if address in file:
                swap_data_list.append(pd.read_csv(os.path.join(self.default_dir, file)))

        return pd.concat(swap_data_list)

    def get_swap_data_by_blocks(self, address, start_block, end_block, save=True):
        """
        Downloads swap event data for a given block interval and saves to csv if required.

        :param address: (str) Contract address of uniswap pool.
        :param start_block: (int) Start block of desired data.
        :param end_block: (int) End block of desired data.
        :param save: (bool) Whether to save output to csv or not.

        :return: (DataFrame) DataFrame of all swap event data
        """

        swap_query = f"""
                select * from `blockchain-etl.ethereum_uniswap.UniswapV3Pool_event_Swap`
                where contract_address='{address}' and block_number > {start_block} and block_number < {end_block}
                order by block_number;
                """
        query_job = self.client.query(swap_query)
        output = query_job.result().to_dataframe()

        if save:
            output.to_csv(f'{self.default_dir}/{address}_{start_block}_{end_block}.csv', index=False)

        return output

    def get_gas_data(self, start_block, end_block, save=True):
        """
        Downloads gas price data for the given blocks.

        :param start_block: (int) Start block of data request.
        :param end_block: (int) End block of data request.
        :param save: (bool) Whether to save output locally.

        :return: (DataFrame) DataFrame of all gas data
        """

        file_name = f'{self.default_dir}/gas-{start_block}-{end_block}.csv'
        if file_name in os.listdir(self.default_dir):
            return pd.read_csv(file_name)
        else:
            # query the data
            gas_query = f"""
                select avg(gas_price) as gas_price, block_number from `bigquery-public-data.crypto_ethereum.transactions`
                where block_number>={start_block} and block_number <{end_block}
                group by block_number
                order by block_number desc;
                """
            query_job = self.client.query(gas_query)
            output = query_job.result().to_dataframe()
            output.to_csv(file_name, index=False)
            return output


class UniswapV3DataL2:
    """
    Class for the extraction of blockdata using paid access to Alchemy archive nodes.
    """

    def __init__(self, path, chain, pool_address, abi_path, api_key):
        """
        Class constructor.
        """

        self.path = path
        self.chain = chain
        self.pool_address = pool_address
        self.abi_path = abi_path
        self.api_key = api_key
        self.endpoint = self.get_end_point()
        self.w3 = self.get_web3()
        self.max_blocks_per_query = self.get_max_blocks()

    def get_end_point(self):
        """
        Gets RPC endpoint for the chain.
        """

        if self.chain.lower() == 'optimism':
            endpoint = f"https://opt-mainnet.g.alchemy.com/v2/{self.api_key}"
        else:
            raise ValueError('Unknown chain ' + self.chain)

        return endpoint

    def get_web3(self):
        """
        Creates the Web3 object from the endpoint.
        """

        if self.chain.lower() == 'optimism':
            w3_obj = Web3(Web3.HTTPProvider(self.endpoint))
            w3_obj.middleware_onion.inject(geth_poa_middleware, layer=0)
        else:
            raise ValueError('Unknown chain ' + self.chain)

        return w3_obj

    def get_max_blocks(self):
        """
        Defines the max number of blocks to request in a single query.
        """

        if self.chain.lower() == 'optimism':
            max_blocks_per_query = 300000
        else:
            raise ValueError('Unknown chain ' + self.chain)

        return max_blocks_per_query

    def get_swap_data(self, download_latest=False):
        """
        Loads any locally stored datafiles and downloads lastest data if required.

        :param download_latest: (bool) Whether to download latest data or just use local data.

        :return: (DataFrame) DataFrame of all swap event data
        """

        pool_contract = self.w3.eth.contract(address=self.pool_address, abi=json.load(open(self.abi_path, 'r')))

        # check if file already exists
        files = os.listdir(self.path)
        end_block = 0
        current_block = 0
        for file in files:
            if self.pool_address in file:
                _, _, _, current_block = file.split('_')
                current_block = int(current_block[:-4])
                end_block = max(current_block, end_block)

        if download_latest:
            output = []

            start_block = end_block
            latest_block = self.w3.eth.block_number

            intermediate_block = start_block + self.max_blocks_per_query

            while start_block < latest_block:
                print("Block number start:" + start_block)
                print("Block number end:" + intermediate_block)
                swap_events = pool_contract.events.Swap.createFilter(fromBlock=start_block, toBlock=intermediate_block)
                try:

                    for event in swap_events.get_all_entries():
                        final_dict = {'block_number': event['blockNumber'], 'txn_hash': event['transactionHash'].hex(),
                                      'amount0': event['args']['amount0'], 'amount1': event['args']['amount1'],
                                      'liquidity': event['args']['liquidity'],
                                      'sqrt_price_x96': event['args']['sqrtPriceX96'],
                                      'tick': event['args']['tick'],
                                      'timestamp': self.w3.eth.getBlock(event['blockNumber'])['timestamp']}
                        # need to speed up block_timestamp
                        output.append(final_dict)

                    start_block = intermediate_block
                    intermediate_block = start_block + self.max_blocks_per_query

                except ValueError as ve:
                    message = ve.args[0]['message']
                    start_str = message.find('[')
                    end_str = message.find(']')
                    start_end = message[start_str + 1:end_str]
                    end_block_str = start_end[start_end.find(',') + 2:]
                    intermediate_block = int(end_block_str, 16)

            data = pd.DataFrame(output)
            data.to_csv(
                f'{self.path}/{self.chain}_{self.pool_address}_{data.block_number.iloc[0]}_{data.block_number.iloc[-1]}.csv',
                index=False)

            # check if directory again exists
        files = os.listdir(self.path)
        swap_data_list = []
        for file in files:
            if self.pool_address in file:
                swap_data_list.append(pd.read_csv(os.path.join(self.path, file)))

        return pd.concat(swap_data_list).sort_values('block_number')
