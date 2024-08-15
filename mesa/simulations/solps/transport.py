from numpy import ndarray


def write_solps_transport_inputfile(
    filename: str,
    grid_dperp: ndarray,
    values_dperp: ndarray,
    grid_chieperp: ndarray,
    values_chieperp: ndarray,
    grid_chiiperp: ndarray,
    values_chiiperp: ndarray,
    set_ana_visc_dperp=True,
    no_pflux=True,
    no_div=False
):
    """
    Writes a b2.transport.inputfile to prepare a SOLPS-ITER run.
    Inputs:
        - filename: name of file to output
        - n_species: number of species to apply transport coefficients to
        - grid_dperp: grid where profile of anomalous radial particle diffusion is specified.
        - values_dperp: profile of anomalous radial particle diffusion
        - grid_chieperp: grid where profile of anomalous radial electron heat diffusion is specified.
        - values_chieperp: profile of anomalous radial electron heat diffusion.
        - grid_chiiperp: grid where profile of anomalous radial ion heat diffusion is specified.
        - values_chiiperp: profile of anomalous raidal ion heat diffusion
        - set_ana_visc_dperp: if true, sets profile of anomalous viscosity to be equal to the anomalous particle diffusion.
        - no_pflux: if true, anomalous transport profiles do not apply in the private flux region(s)
        - no_div: if true, anomalous transport profiles do not apply in the divertor(s)
    """

    # Define list to write the file lines to
    sout = [' &TRANSPORT']

    def build_profile(grid, values, code):
        strings = [f' ndata(1, {code} , 1 )= {len(grid)} ,']
        for i, (x, y) in enumerate(zip(grid, values)):
            line = f' tdata(1,{i + 1:2.0f} , {code} , 1 )= {x:6.6f} , tdata(2,{i + 1:2.0f} , {code} , 1 )= {y:6.6f} ,'
            strings.append(line)
        return strings

    sout.extend(  # Write the anomalous radial particle diffusivity profile
        build_profile(grid_dperp, values_dperp, code=1)
    )

    if set_ana_visc_dperp:
        sout.extend(  # If requested, write the anomalous viscosity
            build_profile(grid_dperp, values_dperp, code=7)
        )

    sout.extend(  # Write the anomalous radial electron heat diffusivity profile
        build_profile(grid_chieperp, values_chieperp, code=3)
    )

    sout.extend(  # Write the anomalous radial ion heat diffusivity profile
        build_profile(grid_chiiperp, values_chiiperp, code=4)
    )

    # TODO - check with james whether '9' is intentionally missing here
    extra_species = [3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16]
    sout.extend([f' addspec({i},{j},1)={i} ,' for i in extra_species for j in [1, 3]])

    if no_pflux:
        sout.append(' no_pflux=.true. ')

    if no_div:
        sout.append(' no_div=.true. ')

    sout.append(' /')

    # Write the list to a file
    with open(filename, 'w') as f:
        for item in sout:
            f.write("%s\n" % item)
