
import os
import logging

from obspy import UTCDateTime
from robustraqn.utils.bayesloc import update_cat_from_bayesloc

# %% TEST TEST TEST
if __name__ == "__main__":
    Logger = logging.getLogger(__name__)
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")

    from obspy.io.nordic.core import read_nordic
    catalog = read_nordic(
        '/home/seismo/WOR/felix/R/SEI/REA/INTEU/2020/12/14-1935-58R.S202012')
    # '/home/seismo/WOR/felix/R/SEI/REA/INTEU/2021/07/05-0423-41R.S202107')
    # 'a8598f5d-d25d-4b69-8547-298589d29bc3'

    Logger.info('Updating catalog from bayesloc solutions')
    top_path = '/home/felix/Documents2/Ridge/Relocation/Bayesloc/'
    bayesloc_path = [
        os.path.join(top_path, 'Ridge_INTEU_09a_oceanic_09'),
        os.path.join(top_path, 'Ridge_INTEU_09a_oceanic_10b')]

    update_cat_from_bayesloc(
        catalog, bayesloc_path, custom_epoch=UTCDateTime(1960, 1, 1, 0, 0, 0),
        agency_id='BER', find_event_without_id=True, s_diff=3,
        max_bayes_error_km=100, add_arrivals=True, update_phase_hints=True,
        keep_best_fit_pick_only=True, remove_1_suffix=True,
        min_phase_probability=0)
