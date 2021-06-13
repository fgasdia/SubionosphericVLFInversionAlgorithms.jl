using Plots

using SubionosphericVLFInversionAlgorithms

function gasparicohn()
    z = 0:2000e3
    c = 1000e3
    gc = gaspari_cohn99_410(z, c)

    plot(z, gc)
end
