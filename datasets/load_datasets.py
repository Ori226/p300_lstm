from keras.utils.data_utils import get_file

def download_and_cache_file(subject_id, experiment_suffix="Color116ms"):
    if subject_id == 'fat':
        url_format = "http://doc.ml.tu-berlin.de/bci/doron/VPfat_11_01_24/RSVP_{}VP{}.mat"
    elif subject_id == 'gcb':
        url_format = "http://doc.ml.tu-berlin.de/bci/doron/VPgcb_11_02_02/RSVP_{}VP{}.mat"
    elif subject_id == 'gcc':
        url_format = "http://doc.ml.tu-berlin.de/bci/doron/VPgcc_11_02_04/RSVP_{}VP{}.mat"
    elif subject_id == 'gcd':
        url_format = "http://doc.ml.tu-berlin.de/bci/doron/VPgcd_11_02_07/RSVP_{}VP{}.mat"

    elif subject_id == 'gce':
        url_format = "http://doc.ml.tu-berlin.de/bci/doron/VPgce_11_02_08/RSVP_{}VP{}.mat"
    elif subject_id == 'gcf':
        url_format = "http://doc.ml.tu-berlin.de/bci/doron/VPgcf_11_02_09/RSVP_{}VP{}.mat"
    elif subject_id == 'gcg':
        url_format = "http://doc.ml.tu-berlin.de/bci/doron/VPgcg_11_02_10/RSVP_{}VP{}.mat"
    elif subject_id == 'gch':
        url_format = "http://doc.ml.tu-berlin.de/bci/doron/VPgch_11_02_11/RSVP_{}VP{}.mat"

    elif subject_id == 'iay':
        url_format = "http://doc.ml.tu-berlin.de/bci/doron/VPiay_11_02_01/RSVP_{}VP{}.mat"

    elif subject_id == 'icn':
        url_format = "http://doc.ml.tu-berlin.de/bci/doron/VPicn_11_02_21/RSVP_{}VP{}.mat"

    elif subject_id == 'icr':
        url_format = "http://doc.ml.tu-berlin.de/bci/doron/VPicr_11_03_03/RSVP_{}VP{}.mat"
    elif subject_id == 'pia':
        url_format = "http://doc.ml.tu-berlin.de/bci/doron/VPpia_11_03_17/RSVP_{}VP{}.mat"
    else:
        raise Exception("subject not familiar")
    url = url_format.format(experiment_suffix, subject_id)

    file_name = url.split("/")[-1]
    return get_file(file_name, url, cache_subdir="p300_lstm")




