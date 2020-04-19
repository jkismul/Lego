import LFPy
import numpy as np
import matplotlib.pyplot as plt
import sys
import neuron

# Handling number of cells to average over
if len(sys.argv)>1:
    iters = int(sys.argv[1])
else:
    iters = 1

# Set morphology to Mainen-Sejnowski L5 pyramidal
morphology = 'morphologies/L5_Mainen96_LFPy.hoc'

# Function for insertine synapses on cell
def insert_synapses(synparams, section, n):
    if section == 'dend':
        maxim = -50
        minim=-1000
    if section == 'apic':
        maxim=1000
        minim=500
    if section=='allsec':
        maxim=1000
        minim=-1000
    '''find n compartments to insert synapses onto'''
    idx = cell.get_rand_idx_area_norm(section=section, nidx=n,z_min=minim,z_max=maxim)

    #Insert synapses in an iterative fashion
    for i in idx:
        synparams.update({'idx' : int(i)})

        # Create synapse(s) and setting times using the Synapse class in LFPy
        s = LFPy.Synapse(cell, **synparams)
        s.set_spike_times(np.array([20.]))

# Define cell parameters
cell_parameters = {
    'morphology': morphology,
    'cm': 1.0,  # membrane capacitance
    'Ra': 150.,  # axial resistance
    'v_init': -65.,  # initial crossmembrane potential
    'passive': True,  # turn on NEURONs passive mechanism for all sections
    'passive_parameters': {'g_pas': 1. / 30000, 'e_pas': -65},
    'nsegs_method': 'lambda_f',  # spatial discretization method
    'lambda_f': 100.,  # frequency where length constants are computed
    'dt': 2. ** -3,  # simulation time step size
    'tstart': -200.,  # start time of simulation, recorders start at t=0
    'tstop': 100.,  # stop simulation at 100 ms.
}
# Define synapse parameters
synapse_parameters = {
    'idx':[],
    'e': 0,  # reversal potential
    'syntype': 'ExpSyn',  # synapse type
    'tau': 10.,  # synaptic time constant
    'weight': .001,  # synaptic weight
    'record_current': True,  # record synapse current
}
# Create a grid of measurement locations, in (mum, micro-meters)
X, Z = np.mgrid[-700:701:50, -400:1201:50]
Y = np.zeros(X.shape)

# Define electrode parameters
grid_electrode_parameters = {
    'sigma': 0.3,  # extracellular conductivity
    'x': X.flatten(),  # electrode requires 1d vector of positions
    'y': Y.flatten(),
    'z': Z.flatten()
}

# Run simulation, electrode object argument in cell.simulate
gr = [] # For storing LFP values in grid of measurement locations
syn_pos = [] # Array of synapse positions
print("running simulation...")

n_syn=200 # Number of synapses per cell
for j in range(iters):
    print(j) # Print progress
    np.random.seed(j+10)
    for i in range(6):
        # Create cell
        cell = LFPy.Cell(**cell_parameters)
        # cell.set_rotation(x=4.99, y=-4.33, z=3.14) # "standard" Mainen-Sejnowsi pyramidal L5 rotation
        cell.set_rotation(x=4.99, y=-4.33) # Align cell so apical points upwards.

        if i > 2:
            synapse_parameters['e'] = -110 # Usually -80, but exaggerated for clarity of plot
        else:
            synapse_parameters['e'] = 0
        if i in [0, 3]:
            np.random.seed(1234+j)
            insert_synapses(synapse_parameters,'dend',n_syn)
        if i in [1,4]:
            np.random.seed(1234+j)
            insert_synapses(synapse_parameters,'apic',n_syn)
        else:
            np.random.seed(1234+j)
            insert_synapses(synapse_parameters,'allsec',n_syn)



        synapse = LFPy.Synapse(cell, **synapse_parameters)
        synapse.set_spike_times(np.array([20.]))
        cell.simulate(rec_imem=True,rec_vmem=True)

        # # Create electrode objects
        grid_electrode = LFPy.RecExtElectrode(cell, **grid_electrode_parameters)
        # # Calculate LFPs
        grid_electrode.calc_lfp()
        gr.append(grid_electrode)
        syn_pos.append([cell.xmid[cell.synidx], cell.zmid[cell.synidx]])

print("done")

# create a plot
fig = plt.figure(figsize=(10,10))
ax = []
mul = []
tit = ['Basal', 'Apical','All', 'Basal', 'Apical', 'All']

csd = []
for i in range(6):
    ax.append(fig.add_subplot(2, 3, i + 1, frameon=False))  # , aspect='equal', frameon=False,label='{}'.format(i)))


    t_show = 25  # time point to show LFP
    tidx = np.where(cell.tvec == t_show)

    # This is the extracellular potential, reshaped to the X, Z
    LFP = (gr[i].LFP[:, tidx]).reshape(X.shape)

    #mean LFP over all iterations
    for j in range(1,iters):
        LFP += (gr[i+j*6].LFP[:,tidx]).reshape(X.shape)
    LFP = LFP/iters


    tst = np.arcsinh(LFP)
    # Separate pos from neg to take log.
    # indx=np.where(LFP<=0)
    # LFP = np.abs(LFP)

    # oners = np.ones(X.shape)
    # mul.append(np.copysign(oners, np.max(gr[i].LFP, 1).reshape(X.shape)))

    # img = np.log10(LFP)
    # img[indx]=np.negative(img[indx])
    #
    # img[np.isnan(img)]=-9.5
    # img[np.isinf(img)]=-9.5
    img=tst
    # print(np.min(img))

    # img[indx]=np.negative(img[indx])

    im = plt.contourf(X, Z, img,
                        cmap='bwr',
                        levels=100,
                        zorder=-2,
                      vmin=-0.0005,
                      vmax=0.0005,
                      # vmid=0.0,
                      # norm=AsinhNorm()
                        # vmin=-0.00011,
                        # vmax=0.00011
                       )

    # im = ax[i].contour(X, Z, img,
    #                     cmap='bwr',
    #                     levels=100,
    #                     zorder=-2,
    #                     linewidths=0.4,
    #                    # vmin=-1e-7,
    #                    # vmax=1e-7,
    #                    vmin=-0.00011,
    #                    vmax=0.00011
    #                    )

    ax[i].set_xticks([])
    ax[i].set_yticks([])
    ax[i].set_title(tit[i])
    # ax[i].set_clim(vmin=-1,vmax=1)
    left, right = ax[i].get_xlim()
    bot,top=ax[i].get_ylim()
    # ax[i].plot([right - 30, right - 30], [bot+100, bot+3500+100], 'k', lw=3, clip_on=False)
    # ax[i].text(right-1250, bot+600, r'3500$\mu$m', rotation=90)
    # ax[i].plot([right - 3500-100, right-100], [bot, bot], 'k', lw=3, clip_on=False)


    a, b = np.float(np.float(syn_pos[i][0][0])), np.float(syn_pos[i][1][0])

    if i > 2:
        # colo = 'gold'
        colo = 'aqua'
    else:
        colo = 'lime'
    # for sec in neuron.h.allsec():
    #     idx = cell.get_idx(sec.name())
    #     ax[i].plot(np.r_[cell.xstart[idx], cell.xend[idx][-1]],
    #                np.r_[cell.zstart[idx], cell.zend[idx][-1]],
    #                color='gray', lw=.5,alpha=0.5)
    # ax[i].scatter(cell.xmid[0], cell.zmid[0], c='gray',alpha=0.1)

    for j in range(n_syn):
        # ax[i].plot(np.float(syn_pos[i][0][0]), np.float(syn_pos[i][1][j]), 'o', ms=5,
        ax[i].plot(np.float(syn_pos[i][0][j]), np.float(syn_pos[i][1][j]), 'o', ms=3,
                   markeredgecolor='k',
                   markeredgewidth=0.0,
                   markerfacecolor=colo,
                   # alpha=0.3,
                   )
    for sec in neuron.h.allsec():
        idx = cell.get_idx(sec.name())
        ax[i].plot(np.r_[cell.xstart[idx], cell.xend[idx][-1]],
                   np.r_[cell.zstart[idx], cell.zend[idx][-1]],
                   color='k', lw=.5, alpha=0.5)
    ax[i].scatter(cell.xmid[0], cell.zmid[0], c='k', alpha=0.1)


# imf = plt.contourf(X, Z, img,
#                     cmap='bwr',
#                     levels=100,
#                     zorder=-2,
#                     vmin=-0.00011,
#                     vmax=0.00011
#                    )



cax = fig.add_axes([0.3, 0.05, 0.5, 0.02], frameon=False)
cbar = fig.colorbar(im, cax=cax,orientation='horizontal')
# print()
cbar_ticks = cbar.get_ticks()
# cbar.set_ticks(np.sinh(cbar_ticks))
tick_labels = []
for i in cbar_ticks:
    # tick_labels.append(np.format_float_scientific(np.sinh(i)*1e4,precision=1))
    tick_labels.append((np.sinh(i)*1e4).round(2))

    # print(i,np.format_float_scientific(np.sinh(i),precision=1))
# cbar.set_ticklabels(np.format_float_scientific(np.sinh(cbar_ticks)))
cbar.set_ticklabels(tick_labels)
cbar.set_label('$\Phi(\mathbf{r}, t=25ms)$ (nV)')
# cbar.outline.set_visible(False)

fig.text(0.1, 0.65, 'Excitatory', rotation=90,fontsize=12)

fig.text(0.1, 0.2, 'Inhibitory', rotation=90,fontsize=12)

plt.savefig('Lego.pdf', dpi=300)
