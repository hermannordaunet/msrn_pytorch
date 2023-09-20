from datetime import timedelta
from utils.flops_counter import flops_to_string, params_to_string


def print_min_max_conf(min_conf, max_conf, mean_conf, mode="TRAIN", include_last=False):
    min_string = f"[{mode}]: Min exit conf from batch: "
    max_string = f"[{mode}]: Max exit conf from batch: "
    mean_string = f"[{mode}]: Mean exit conf from batch: "

    for exit, (min_value, max_value, mean_value) in enumerate(
        zip(min_conf, max_conf, mean_conf)
    ):
        min_string += f"{exit}: {min_value:.4%}, "
        max_string += f"{exit}: {max_value:.4%}, "
        mean_string += f"{exit}: {mean_value:.4%}, "

    print(min_string[:-2])
    print(max_string[:-2])
    print(mean_string[:-2])


def print_cost_of_exits(model):
    # print cost of exit blocks
    total_flops, total_params = model.complexity[-1]

    for i, (flops, params) in enumerate(model.complexity[:-1]):
        print(
            f"ee-block-{i}: flops={flops_to_string(flops)}, params={params_to_string(params)}, cost-rate={(flops / total_flops):.2f}"
        )
    print(
        f"exit-full-model: flops={flops_to_string(total_flops)}, params={params_to_string(total_params)}, cost-rate={(1.000):.2f}"
    )


def get_time_hh_mm_ss(sec):
    # create timedelta and convert it into string
    td_str = str(timedelta(seconds=sec))

    # split string into individual component
    x = td_str.split(":")
    time_as_string = f"hh:mm:ss: {x[0]} hours, {x[1]} minutes, {x[2]} seconds"

    return time_as_string
