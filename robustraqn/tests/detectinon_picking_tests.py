

# %% ###########################  TEST TEST TEST  #############################
if __name__ == "__main__":
    # Set the path to the folders with continuous data:
    archive_path = '/data/seismo-wav/SLARCHIVE'
    # archive_path2 = '/data/seismo-wav/EIDA/archive'
    client = Client(archive_path)
    # client2 = Client(archive_path2)

    sta_translation_file = "station_code_translation.txt"
    selected_stations = ['ASK','BER','BLS5','DOMB','FOO','HOMB','HYA','KMY',
                        'ODD1','SKAR','SNART','STAV','SUE','KONO','DOMB',
                        #'NAO01','NB201','NBO00','NC204','NC303','NC602',
                        'NAO00','NAO01','NAO02','NAO03','NAO04','NAO05',
                        'NB200','NB201','NB202','NB203','NB204','NB205',
                        'NBO00','NBO01','NBO02','NBO03','NBO04','NBO05',
                        'NC200','NC201','NC202','NC203','NC204','NC205',
                        'NC300','NC301','NC302','NC303','NC304','NC305',
                        'NC400','NC401','NC402','NC403','NC404','NC405',
                        'NC600','NC601','NC602','NC603','NC604','NC605',
                        'STRU']
    # Add some extra stations from Denmark / Germany / Netherlands
    # add_stations =  ['NAO00','NAO02','NAO03','NAO04','NAO05',
    #                 'NB200','NB202','NB203','NB204','NB205',
    #                 'NBO00','NBO01','NBO02','NBO03','NBO04','NBO05',
    #                 'NC200','NC201','NC202','NC203','NC205',
    #                 'NC300','NC301','NC302','NC304','NC305',
    #                 'NC400','NC401','NC402','NC403','NC404','NC405',
    #                 'NC600','NC601','NC603','NC604','NC605']

    relevant_stations = get_all_relevant_stations(
        selected_stations, sta_translation_file=sta_translation_file)
    # add_stations = get_all_relevant_stations(
    #     add_stations, sta_translation_file=sta_translation_file)
    # all_stations = relevant_stations + add_stations

    startday = UTCDateTime(2021,4,1,0,0,0)
    endday = UTCDateTime(2021,4,30,0,0,0)

    inv_file = '~/Documents2/ArrayWork/Inventory/NorSea_inventory.xml'
    inv_file = os.path.expanduser(inv_file)
    inv = get_updated_inventory_with_noise_models(
        os.path.expanduser(inv_file),
        pdf_dir='~/repos/ispaq/WrapperScripts/PDFs/',
        outfile='inv.pickle', check_existing=True)

    template_path ='Templates'
    #template_path='LagCalcTemplates'
    parallel = True
    cores = 40
    # det_folder = 'Detections_onDelta'
    det_folder = 'ReDetections_MAD9'

    remove_response = False
    noise_balancing = False
    check_array_misdetections = False
    write_party = True
    # threshold = 11
    new_threshold = 14
    n_templates_per_run = 30
    min_det_chans = 15
    only_request_detection_stations = True

    # Read templates from file
    Logger.info('Starting template reading')
    tribe = Tribe().read('TemplateObjects/Templates_min21tr_27.tgz')
    Logger.info('Tribe archive readily read in')
    if check_array_misdetections:
        short_tribe = Tribe().read(
            'TemplateObjects/short_Templates_min21tr_27.tgz')
        Logger.info('Short-tribe archive readily read in')
    n_templates = len(tribe)

    #Check templates for duplicate channels
    tribe = check_duplicate_template_channels(tribe)

    # Read in and process the daylong data
    dates = pd.date_range(startday.datetime, endday.datetime, freq='1D')
    # For each day, read in data and run detection from templates
    current_year = None
    for date in dates:
            # Load in Mustang-like ISPAQ stats for the whole year
        if not date.year == current_year:
            current_year = date.year
            ispaq = read_ispaq_stats(folder=
                '/home/felix/repos/ispaq/WrapperScripts/Parquet_database/csv_parquet',
                stations=relevant_stations, startyear=current_year,
                endyear=current_year, ispaq_prefixes=['all'],
                ispaq_suffixes=['simpleMetrics','PSDMetrics'],
                file_type = 'parquet')
        pick_events_for_day(
            date=date, det_folder=det_folder, template_path=template_path,
            ispaq=ispaq, clients=[client], relevant_stations=relevant_stations,
            only_request_detection_stations=only_request_detection_stations,
            noise_balancing=noise_balancing, remove_response=remove_response,
            inv=inv, parallel=parallel, cores=cores,
            check_array_misdetections=check_array_misdetections,
            write_party=write_party, new_threshold=new_threshold,
            n_templates_per_run=n_templates_per_run,
            min_det_chans=min_det_chans)