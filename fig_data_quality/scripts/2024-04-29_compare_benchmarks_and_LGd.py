from data_quality.tables import load_channels, load_clusters
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np

dl = Path("/Users/chris/Downloads")

benchmark_pids = ['1a276285-8b0e-4cc9-9f0a-a3a002978724',
          '1e104bf4-7a24-4624-a5b2-c2c8289c0de7',
          '5d570bf6-a4c6-4bf1-a14b-2c878c84ef0e',
          '5f7766ce-8e2e-410c-9195-6bf089fea4fd',
          '6638cfb3-3831-4fc2-9327-194b76cf22e1',
          '749cb2b7-e57e-4453-a794-f6230e4d0226',
          'd7ec0892-0a6c-4f4f-9d8f-72083692af5c',
          'da8dfec1-d265-44e8-84ce-6ae9c109b8bd',
          'dab512bd-a02d-4c1f-8dbc-9155a163efc0',
          'dc7e9403-19f7-409f-9240-05ee57cb7aea',
          'e8f9fba4-d151-4b00-bee7-447f0f3e752c',
          'eebcaf65-7fa4-4118-869d-a084e84530e2',
          'fe380793-8035-414e-b000-09bfe5ece92a']

lg_pids = ["099c0519-640b-4eb7-867c-998dc337579d",
        "0eb65305-bb95-4bf1-a154-1b810c0cff25",
        "11a5a93e-58a9-4ed0-995e-52279ec16b98",
        "1a924329-65aa-465d-b201-c2dd898aebd0",
        "1e176f17-d00f-49bb-87ff-26d237b525f1",
        "220bca21-4cf8-43f1-a213-71645899c571",
        "31811232-52ef-456c-9772-5021c00b2bc9",
        "31f0400a-71f5-436c-ada4-dce50627dc73",
        "501ab543-9d02-4085-956d-22b9b3eeb543",
        "56f2a378-78d2-4132-b3c8-8c1ba82be598",
        "5bfcf68f-4e0b-46f8-a9e3-8e919d5ddd1c",
        "5eebb5d6-cf9b-49a0-b297-0701c4d90173",
        "63517fd4-ece1-49eb-9259-371dc30b1dd6",
        "6be21156-33b0-4f70-9a0f-65b3e3cd6d4a",
        "70da415f-444d-4148-ade7-a1f58a16fcf8",
        "7be00744-2e27-4062-8f56-3969e24e9990",
        "82765a73-2a03-42ca-ba98-194247caa62e",
        "8d59da25-3a9c-44be-8b1a-e27cdd39ca34",
        "8fbfa285-d721-4a1a-8c18-f36652c17909",
        "9338f8bb-e097-46d8-81b4-7f37beb4d308",
        "95fd67e6-cbff-4356-80c7-5a03b1bf6b8a",
        "a6fe3779-2b77-4b66-a625-a6078720e412",
        "b8df4cc3-e973-4267-8cf8-a55cea77e1ac",
        "bef05a5c-68c3-4513-87c7-b3151c88da8e",
        "c2363000-27a6-461e-940b-15f681496ed8",
        "c9fb5e2e-bd92-41d8-8b7e-394005860a1e",
        "eb99c2c8-e614-4240-931b-169bed23e9f5",
        "f26a6ab1-7e37-4f8d-bb50-295c056e1062",
       ]

re_clusters = load_clusters("re_147")
re_pids = list(re_clusters.index.get_level_values(0).unique())
re_benchmarks = list(set(re_pids).intersection(set(benchmark_pids)))
re_lgd = list(set(re_pids).intersection(set(lg_pids)))

bwm_clusters = load_clusters("bwm_147")
bwm_pids = list(bwm_clusters.index.get_level_values(0).unique())
bwm_benchmarks = list(set(bwm_pids).intersection(set(benchmark_pids)))
bwm_lgd = list(set(bwm_pids).intersection(set(lg_pids)))

re_channels = load_channels("re")
bwm_channels = load_channels("bwm")
#re_channels = re_channels.loc[list(re_clusters.index.get_level_values(0).unique())]
bwm_channels = bwm_channels.loc[list(bwm_clusters.index.get_level_values(0).unique())]

re_old = load_clusters("re")
bwm_old = load_clusters("bwm")
lgd_old = pd.concat([re_old.loc[re_lgd], bwm_old.loc[bwm_lgd]])

# benchmarks
# the one benchmark pid in RE is also in bwm
benchmarks_147 = bwm_clusters.loc[bwm_benchmarks]

re_lgd_147 = re_clusters.loc[re_lgd]
bwm_lgd_147 = bwm_clusters.loc[bwm_lgd]
lgd_147 = pd.concat([re_lgd_147, bwm_lgd_147])

# yields
yield_benchmarks = {}
yield_old_benchmarks = {}
pct_change_benchmarks = {}

for bpid in bwm_benchmarks:
    npassing = sum(benchmarks_147.loc[bpid].label==1.)
    npassing_old = sum(bwm_old.loc[bpid].label==1.)
    nchan = sum(bwm_channels.loc[bpid].cosmos_acronym != "void")
    yield_new= npassing / nchan
    yield_benchmarks[bpid] = yield_new
    yield_old = npassing_old / nchan
    yield_old_benchmarks[bpid] = yield_old
    pct_change_benchmarks[bpid] = 100 * (yield_new - yield_old) / yield_old


yield_lgd = {}
yield_old_lgd = {}
pct_change_lgd = {}

for lpid in re_lgd:
    npassing = sum(lgd_147.loc[lpid].label==1.)
    npassing_old = sum(re_old.loc[lpid].label==1.)
    nchan = sum(re_channels.loc[lpid].cosmos_acronym != "void")
    yield_new= npassing / nchan
    yield_lgd[lpid] = yield_new
    yield_old = npassing_old / nchan
    yield_old_lgd[lpid] = yield_old
    pct_change_lgd[lpid] = 100 * (yield_new - yield_old) / yield_old

for lpid in bwm_lgd:
    npassing = sum(lgd_147.loc[lpid].label==1.)
    npassing_old = sum(bwm_old.loc[lpid].label==1.)
    nchan = sum(bwm_channels.loc[lpid].cosmos_acronym != "void")
    yield_new= npassing / nchan
    yield_lgd[lpid] = yield_new
    yield_old = npassing_old / nchan
    yield_old_lgd[lpid] = yield_old
    pct_change_lgd[lpid] = 100 * (yield_new - yield_old) / yield_old

dir160 = dl.joinpath("benchmark_160")

dfs160 = {}
for pid in bwm_benchmarks:
    dfs160[pid] = pd.read_parquet(dir160.joinpath(f"{pid}_clusters.pqt"))
benchmarks160 = pd.concat(dfs160)

yield_160_benchmarks = {}
pct_change160_benchmarks = {}
for pid in bwm_benchmarks:
    npassing = sum(benchmarks160.loc[pid].label==1.)
    nchan = sum(bwm_channels.loc[pid].cosmos_acronym != "void")
    yield_160 = npassing / nchan
    yield_old = yield_old_benchmarks[pid]
    yield_160_benchmarks[pid] = yield_160
    pct_change160_benchmarks[pid] = 100 * (yield_160 - yield_old) / yield_old

fig, ax = plt.subplots(1, 2)
for i, pid in enumerate(bwm_benchmarks):
    ax[i].bar(["original", "1.4.7", "1.6.0"], 
            [yield_old_benchmarks[pid],
            yield_benchmarks[pid],
            yield_160_benchmarks[pid]],color=['b', 'r', 'orange'])
    ax[i].set_title(f"{pid[:4]}")
    ax[i].set_ylim(0, 0.7)
ax[0].set_ylabel("yield")
fig.suptitle("Yield across versions: benchmark insertions")
fig.tight_layout()

fig, ax = plt.subplots(1, 2)
for i, pid in enumerate(bwm_benchmarks):
    ax[i].bar(["original", "1.4.7", "1.6.0"], 
            [0,
            pct_change_benchmarks[pid],
            pct_change160_benchmarks[pid]],color="lightblue")
    ax[i].set_title(f"{pid[:4]}")
    ax[i].set_ylim(-50, 100)
ax[0].set_ylabel(f"% yield change")
fig.suptitle(f"% change in yield: LGd insertions")
fig.tight_layout()

nbins = 40
bins = np.logspace(0, np.log10(1000), num=nbins)
hist_kwargs = {"bins":bins, "histtype":"step"}

for i, pid in enumerate(bwm_benchmarks):
    table_old = bwm_old.loc[pid].copy()
    table_147 = bwm_clusters.loc[pid].copy()
    table_160 = benchmarks160.loc[pid].copy()

    table_old["amp_median"] *= 1e6
    table_147["amp_median"] *= 1e6
    table_160["amp_median"] *= 1e6

    table_old_p = table_old[table_old.label==1.0]
    table_147_p = table_147[table_147.label==1.0]
    table_160_p = table_160[table_160.label==1.0]

    fig, ax = plt.subplots(1, 2, figsize=(9, 4))

    

    table_old["amp_median"].hist(ax=ax[0], linestyle="dashed", edgecolor="grey", label="OG-all", **hist_kwargs)
    table_old_p["amp_median"].hist(ax=ax[0], edgecolor="grey", label="OG-passing",**hist_kwargs)
    table_147["amp_median"].hist(ax=ax[0], linestyle="dashed", edgecolor="red", label=f"v1.4.7-all", **hist_kwargs)
    table_147_p["amp_median"].hist(ax=ax[0], edgecolor="red", label=f"v1.4.7-passing", **hist_kwargs)

    table_old["amp_median"].hist(ax=ax[1], linestyle="dashed", edgecolor="grey", label="OG-all", **hist_kwargs)
    table_old_p["amp_median"].hist(ax=ax[1], edgecolor="grey", label="OG-passing",**hist_kwargs)
    table_160["amp_median"].hist(ax=ax[1], linestyle="dashed", edgecolor="orange", label=f"v1.6.0-all", **hist_kwargs)
    table_160_p["amp_median"].hist(ax=ax[1], edgecolor="orange", label=f"v1.6.0-passing", **hist_kwargs)

    ax[0].grid(False)
    ax[1].grid(False)

    ax[0].set_xscale("log")
    ax[1].set_xscale("log")

    ax[0].set_ylim(0, 150)
    ax[1].set_ylim(0, 150)

    ax[0].set_xlim(1, 1000)
    ax[1].set_xlim(1, 1000)

    ax[0].set_title("original vs. 1.4.7")
    ax[1].set_title("original vs. 1.6.0")

    ax[0].vlines(50,
        ax[0].get_ylim()[0],
        ax[0].get_ylim()[1],
        linestyles="dashed",
        label="50 uV",
        color="black",
        linewidths=1.0,)
    ax[1].vlines(50,
        ax[0].get_ylim()[0],
        ax[0].get_ylim()[1],
        linestyles="dashed",
        label="50 uV",
        color="black",
        linewidths=1.0,)
    
    ax[0].legend()
    ax[1].legend()
    ax[0].set_xlabel("log(amp_median), uV")
    ax[1].set_xlabel("log(amp_median), uV")
    ax[0].set_ylabel("# units")
    ax[1].set_ylabel("# units")


    fig.suptitle(f"{pid} amplitude distribution\n")
    fig.tight_layout()

lgd_160 = pd.read_parquet(dl.joinpath("LGd_metrics_pyks_1.6.00.pqt")).loc[re_lgd + bwm_lgd]

# LGd
all_lgd = re_lgd + bwm_lgd

# yield
fig, ax = plt.subplots(3, 3)
for i, _ax in enumerate(ax.flat):
    pid = all_lgd[i]
    npassing_old = sum(lgd_old.loc[pid].label==1.)
    npassing_147 = sum(lgd_147.loc[pid].label==1.)
    npassing_160 = sum(lgd_160.loc[pid].label==1.)
    if pid in re_lgd:
        nchan = sum(re_channels.loc[pid].cosmos_acronym!="void")
    else:
        nchan = sum(bwm_channels.loc[pid].cosmos_acronym!="void")
    yield_old = npassing_old / nchan
    yield_147 = npassing_147 / nchan
    yield_160 = npassing_160 / nchan
    _ax.bar(["Old", "1.4.7", "1.6.0"], [yield_old, yield_147, yield_160], 
            color=["b", "r", "orange"])
    _ax.set_title(f"{pid[:4]}")
    _ax.set_ylim(0.0, 0.6)
for i in range(3):
    ax[i, 0].set_ylabel("yield")
fig.suptitle(f"Yield across versions: LGd insertions")
fig.tight_layout()

# % change yield
fig, ax = plt.subplots(3, 3)
for i, _ax in enumerate(ax.flat):
    pid = all_lgd[i]
    npassing_old = sum(lgd_old.loc[pid].label==1.)
    npassing_147 = sum(lgd_147.loc[pid].label==1.)
    npassing_160 = sum(lgd_160.loc[pid].label==1.)
    if pid in re_lgd:
        nchan = sum(re_channels.loc[pid].cosmos_acronym!="void")
    else:
        nchan = sum(bwm_channels.loc[pid].cosmos_acronym!="void")
    yield_old = npassing_old / nchan
    yield_147 = npassing_147 / nchan
    yield_160 = npassing_160 / nchan
    _ax.bar(["Old", "1.4.7", "1.6.0"], [0, 100*(yield_147-yield_old)/yield_old,  100*(yield_160-yield_old)/yield_old], 
            color="lightblue")
    _ax.set_title(f"{pid[:4]}")
    _ax.set_ylim(-50, 100)
for i in range(3):
    ax[i, 0].set_ylabel("% yield change")
fig.suptitle(f"% change in yield: LGd insertions")
fig.tight_layout()

# amp dist for LGd
table_old = lgd_old.copy()
table_147 = lgd_147.copy()
table_160 = lgd_160.copy()

table_old["amp_median"] *= 1e6
table_147["amp_median"] *= 1e6
table_160["amp_median"] *= 1e6

table_old_p = table_old[table_old.label==1.0]
table_147_p = table_147[table_147.label==1.0]
table_160_p = table_160[table_160.label==1.0]

fig, ax = plt.subplots(1, 2, figsize=(9, 4))

table_old["amp_median"].hist(ax=ax[0], linestyle="dashed", edgecolor="grey", label="OG-all", **hist_kwargs)
table_old_p["amp_median"].hist(ax=ax[0], edgecolor="grey", label="OG-passing",**hist_kwargs)
table_147["amp_median"].hist(ax=ax[0], linestyle="dashed", edgecolor="red", label=f"v1.4.7-all", **hist_kwargs)
table_147_p["amp_median"].hist(ax=ax[0], edgecolor="red", label=f"v1.4.7-passing", **hist_kwargs)

table_old["amp_median"].hist(ax=ax[1], linestyle="dashed", edgecolor="grey", label="OG-all", **hist_kwargs)
table_old_p["amp_median"].hist(ax=ax[1], edgecolor="grey", label="OG-passing",**hist_kwargs)
table_160["amp_median"].hist(ax=ax[1], linestyle="dashed", edgecolor="orange", label=f"v1.6.0-all", **hist_kwargs)
table_160_p["amp_median"].hist(ax=ax[1], edgecolor="orange", label=f"v1.6.0-passing", **hist_kwargs)

ax[0].grid(False)
ax[1].grid(False)

ax[0].set_xscale("log")
ax[1].set_xscale("log")

ax[0].set_ylim(0, 800)
ax[1].set_ylim(0, 800)

ax[0].set_xlim(1, 1000)
ax[1].set_xlim(1, 1000)

ax[0].set_title("original vs. 1.4.7")
ax[1].set_title("original vs. 1.6.0")

ax[0].vlines(50,
    ax[0].get_ylim()[0],
    ax[0].get_ylim()[1],
    linestyles="dashed",
    label="50 uV",
    color="black",
    linewidths=1.0,)
ax[1].vlines(50,
    ax[0].get_ylim()[0],
    ax[0].get_ylim()[1],
    linestyles="dashed",
    label="50 uV",
    color="black",
    linewidths=1.0,)

ax[0].legend()
ax[1].legend()
ax[0].set_xlabel("log(amp_median), uV")
ax[1].set_xlabel("log(amp_median), uV")
ax[0].set_ylabel("# units")
ax[1].set_ylabel("# units")

fig.suptitle(f"Amplitude distribution: LGd insertions\n")
fig.tight_layout()