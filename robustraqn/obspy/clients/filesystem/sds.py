from obspy.core import Stream

import logging
Logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format=("%(asctime)s\t%(name)40s:%(lineno)s\t%(funcName)20s()\t%" +
            "(levelname)s\t%(message)s"))
"[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"

from signal import signal, SIGSEGV
from timeit import default_timer

from joblib import Parallel, delayed
from multiprocessing import Pool, cpu_count
from multiprocessing.pool import ThreadPool
from eqcorrscan.utils.correlate import pool_boy
from obspy.clients.filesystem.sds import Client


def _get_waveforms_bulk(client, bulk):
    """
    """
    st = Stream()
    for arg in bulk:
        try:
            st += client.get_waveforms(*arg)
        except Exception as e:
            document_client_read_error(e)
            continue
    return st


def get_waveforms_bulk(self, bulk, parallel=False, cores=16):
    """
    Perform a bulk-waveform request in parallel. Return one stream.
    There seems to be a negative effect on speed if there's too many read-
    threads - for now set default to 16.

    :type client: obspy.obspy.Client
    :param name: Client to request the data from.
    :type bulk: list of tuples
    :param bulk: Information about the requested data.
    :type parallel: book
    :param parallel: Whether to run reading of waveform files in parallel.
    :type cores: int
    :param bulk: Number of parallel processors to use.

    :rtype: obspy.core.Stream
    """
    # signal(SIGSEGV, sigsegv_handler)
    Logger.info('Start bulk-read')
    outtic = default_timer()
    st = Stream()
    if parallel:
        if cores is None:
            cores = min(len(bulk), cpu_count())

        # Switch to Process Pool for now. Threadpool is quicker, but it is not
        # stable against errors that need to be caught in Mseed-library. Then
        # as segmentation fault can crash the whole program. ProcessPool needs
        # to be spawned, otherwise this can cause a deadlock.

        # Logger.info('Start bulk-read paralle pool')
        # Process / spawn handles segmentation fault better?
        #with pool_boy(Pool=ThreadPool, traces=len(bulk), cores=cores) as pool:
        # with pool_boy(Pool=Pool, traces=len(bulk), cores=cores) as pool:

        # with pool_boy(Pool=get_context("spawn").Pool, traces=len(bulk),
        #               cores=cores) as pool:
        #     results = [pool.apply_async(
        #         _get_waveforms_bulk, args=(client, [arg])) for arg in bulk]

        # Use joblib with loky-pools; this is the most stable
        results = Parallel(n_jobs=cores)(
            delayed(_get_waveforms_bulk)(self, [arg]) for arg in bulk)
        # Concatenate all NSLC-streams into one stream
        st = Stream()
        for res in results:
            # st += res.get()
            try:
                st += res
            except Exception as e:
                Logger.error(e, exc_info=True)
        # pool.close()
        # pool.join()
        # pool.terminate()
    else:
        st = _get_waveforms_bulk(self, bulk)

    outtoc = default_timer()
    Logger.info('Bulk-reading of waveforms took: {0:.4f}s'.format(
        outtoc - outtic))
    return st


def document_client_read_error(s):
    """
    Function to be called when an exception occurs within one worker in
        `get_waveforms_bulk_parallel`.

    :type s: str
    :param s: Error message
    """
    Logger.error("Error reading waveform file - skipping this file.")
    Logger.error(s, exc_info=True)


# def sigsegv_handler(sigNum, frame):
#     """
#     Segmentation fault handler (can occur in MSEED-read)
#     """
#     print("handle signal", sigNum)

def _get_waveforms_bulk_parallel_naive(self, bulk, parallel=True,
                                           cores=None):
        """
        parallel implementation of get_waveforms_bulk.
        """
        # signal.signal(signal.SIGSEGV, sigsegv_handler)

        outtic = default_timer()
        st = Stream()
        if parallel:
            if cores is None:
                cores = min(len(bulk), cpu_count())
            # There seems to be a negative effect on speed if there's too many
            # read-threads - For now set limit to 16
            cores = min(cores, 16)

            # Logger.info('Start bulk-read paralle pool')
            with pool_boy(
                    Pool=ThreadPool, traces=len(bulk), cores=cores) as pool:
                    # Pool=Pool, traces=len(bulk), cores=cores) as pool:
                results = [pool.apply_async(
                    self.get_waveforms,
                    args=arg,
                    error_callback=document_client_read_error)
                        for arg in bulk]
            # Need to handle possible read-errors in each request when getting
            # each request-result.
            for res in results:
                try:
                    st += res.get()
                # InternalMSEEDError
                except Exception as e:
                    Logger.error(e)
                    pass
        else:
            for arg in bulk:
                try:
                    st += self.get_waveforms(*arg)
                except Exception as e:
                    document_client_read_error(e)
                    continue
        outtoc = default_timer()
        Logger.info('Bulk-reading of waveforms took: {0:.4f}s'.format(
            outtoc - outtic))
        return st


# def get_parallel_waveform_client(waveform_client):
#     """
#     Bind a `get_waveforms_bulk` method to waveform_client if it doesn't already
#     have one.
#     """
#     def _get_waveforms_bulk_parallel_naive(self, bulk, parallel=True,
#                                            cores=None):
#         """
#         parallel implementation of get_waveforms_bulk.
#         """
#         # signal.signal(signal.SIGSEGV, sigsegv_handler)

#         st = Stream()
#         if parallel:
#             if cores is None:
#                 cores = min(len(bulk), cpu_count())
#             # There seems to be a negative effect on speed if there's too many
#             # read-threads - For now set limit to 16
#             cores = min(cores, 16)

#             # Logger.info('Start bulk-read paralle pool')
#             with pool_boy(
#                     # Pool=ThreadPool, traces=len(bulk), cores=cores) as pool:
#                     Pool=Pool, traces=len(bulk), cores=cores) as pool:
#                 results = [pool.apply_async(
#                     self.get_waveforms,
#                     args=arg,
#                     error_callback=document_client_read_error)
#                         for arg in bulk]
#             # Need to handle possible read-errors in each request when getting
#             # each request-result.
#             for res in results:
#                 try:
#                     st += res.get()
#                 # InternalMSEEDError
#                 except Exception as e:
#                     Logger.error(e)
#                     pass
#         else:
#             for arg in bulk:
#                 try:
#                     st += self.get_waveforms(*arg)
#                 except Exception as e:
#                     document_client_read_error(e)
#                     continue
#         return st

#     # add waveform_bulk method dynamically if it doesn't exist already
#     if not hasattr(waveform_client, "get_waveforms_bulk_parallel"):
#         bound_method = _get_waveforms_bulk_parallel_naive.__get__(
#             waveform_client)
#         setattr(waveform_client, "get_waveforms_bulk_parallel", bound_method)

#     return waveform_client


# Client.get_parallel_waveform_client = get_parallel_waveform_client
# Client = Client.get_parallel_waveform_client()
Client._get_waveforms_bulk_parallel_naive = _get_waveforms_bulk_parallel_naive
Client.get_waveforms_bulk_parallel = _get_waveforms_bulk_parallel_naive
Client.get_waveforms_bulk = get_waveforms_bulk