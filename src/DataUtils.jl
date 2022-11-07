module DataUtils
export get_awgn, get_N0
export add_cartesian_awgn, add_complex_awgn
export get_grid_2d
export get_symbol_error_rate, get_bit_error_rate_si, get_bit_error_rate_sb
export get_random_bits, get_random_data_si, get_random_data_sb, get_all_unique_symbols
export bits_to_symbols, symbols_to_integers, bits_to_integers, integers_to_symbols, symbols_to_bits, integers_to_bits
export complex_to_cartesian, cartesian_to_complex

using Random
using CUDA


"""
N0: Noise power
n: The shape of the tensor returned is [2,n]
Each entry is i.i.d Gaussian with mean 0 and standard deviation sqrt(N0/2)
"""
function get_awgn(N0::Float32, n)
    noise = randn(Float32, (2, n)) .* sqrt(N0 / 2.0f0)
    noise
end


"""
N0: Noise power
n: The shape of the tensor returned is [2,n]
Each entry is i.i.d Gaussian with mean 0 and standard deviation sqrt(N0/2)
"""
function cu_get_awgn(N0::Float32, n)
    noise = CUDA.randn(Float32, (2, n)) .* sqrt(N0 / 2.0f0)
    noise
end


"""
SNR_db: The desired signal to noise ratio in db scale
signal_power: The signal power in linear scale
"""
function get_N0(SNR_db, signal_power)
    SNR = 10f0 ^ (0.1f0 * SNR_db)
    N0 = signal_power / SNR
    N0
end


"""
Inputs:
data_c: tensor of type float and shape (2,n) containing modulated symbols
SNR_db: Desired signal to noise ratio in db
signal_power: Signal power in linear scale (Default = 1.0)
Output:
data_c_noisy: Noisy modulated symbols where noise such that we get desired SNR_db
"""
function add_cartesian_awgn(data_c::Matrix{Float32}, SNR_db::Float32; signal_power=1f0)
    N0 = get_N0(SNR_db, signal_power)
    noise = get_awgn(N0, size(data_c, 2))
    return data_c .+ noise
end

function add_cartesian_awgn(data_c::CuArray{Float32, 2}, SNR_db::Float32; signal_power=1f0)
    N0 = get_N0(SNR_db, signal_power)
    noise = cu_get_awgn(N0, size(data_c, 2))
    return data_c .+ noise
end


"""
Inputs:
data_c: array of type complex and shape (n) containing modulated symbols
SNR_db: Desired signal to noise ratio in db
signal_power: Signal power in linear scale (Default = 1.0)
Output:
data_c_noisy: Noisy modulated symbols where noise such that we get desired SNR_db
"""
add_complex_awgn(data_c::Vector{ComplexF32}, SNR_db::F1; signal_power::F2=1f0) where {F1 <: Real, F2 <: Real} = add_complex_awgn(data_c, Float32(SNR_db), signal_power=Float32(signal_power))

function add_complex_awgn(data_c::Vector{ComplexF32}, SNR_db::Float32; signal_power::Float32=1f0)
    N0 = get_N0(SNR_db, signal_power)
    noise = get_awgn(N0, size(data_c, 1))
    return data_c .+ noise[1, :] .+ 1im .* noise[2, :]
end

function add_complex_awgn(data_c::CuArray{ComplexF32, 1}, SNR_db::Float32; signal_power::Float32=1f0)
    N0 = get_N0(SNR_db, signal_power)
    noise = cu_get_awgn(N0, size(data_c, 1))
    return data_c .+ noise[1, :] .+ 1im .* noise[2, :]
end


function get_grid_2d(grid=[-1.5; 1.5], points_per_dim=100)
    grid_1d = LinRange(grid[1], grid[2], points_per_dim)
    grid_2d = permutedims(hcat(repeat(grid_1d, inner=points_per_dim),
                               repeat(grid_1d, outer=points_per_dim)))
    grid_2d
end


"""
data_si: tensor of shape [n]
labels_si_g: tensor of shape [n]
Returns the number of indices where these differ divided by n
"""
function get_symbol_error_rate(data_si, labels_si_g)
    @assert length(size(data_si)) == length(size(labels_si_g))
    errs = sum(labels_si_g != data_si)
    errs / length(data_si)
end


"""
data_si: array of shape [n]
labels_si_g: array of shape [n]
bits_per_symbol: integer corresponding to number of bits per symbol
Returns the number of bit errors divided by n
"""
function get_bit_error_rate_si(data_si, labels_si_g, bits_per_symbol)
    data_sb = integers_to_symbols(data_si, bits_per_symbol)
    labels_sb_g = integers_to_symbols(labels_si_g, bits_per_symbol)
    get_bit_error_rate_sb(data_sb, labels_sb_g)
end


"""
data_sb: array of shape [n]
labels_sb_g: array of shape [n]
Returns the number of bit errors divided by n
"""
function get_bit_error_rate_sb(data_sb, labels_sb_g)
    errs = sum(data_sb .!= labels_sb_g)
    bit_error_rate = errs / length(data_sb)
    bit_error_rate
end


"""
Return integer array of 0-1 of shape [n]
"""
function get_random_bits(n; cuda::Bool=false)
    data = rand(UInt16[0 1], (n,))
    cuda ? cu(data) : data
end


"""
Generate random data for integer representation of symbols between [0, 2**bits_per_symbol]
shape [n] --> n random symbols = n*bits_per_symbol random bits
"""
function get_random_data_si(n, bits_per_symbol; cuda::Bool=false)
    data = rand(0:2 ^ bits_per_symbol - 1, (n,))
    cuda ? cu(data) : data
end


"""
Generate random data for bit representation of symbols between [0, 2**bits_per_symbol]
shape [bits_per_symbol x n] --> n random symbols = n*bits_per_symbol random bits
"""
function get_random_data_sb(n, bits_per_symbol; cuda::Bool=false)
    data = rand(UInt16[0 1], (bits_per_symbol, n))
    cuda ? cu(data) : data
end


"""
Generate all unique symbols for bit_per_symbol
shape [bits_per_symbol x 2^bits_per_symbol] --> 2^bits_per_symbol symbols
"""
function get_all_unique_symbols(bits_per_symbol)
    all_symbs = collect(0:2^bits_per_symbol-1)
    integers_to_symbols(all_symbs, bits_per_symbol)
end


"""
Converts array of integer representation of bits to an array of bits
Inputs:
data_si: array of type integer containing integer representation of rows of data_sb,
         of shape [m]
bits_per_symbol: scalar
Output:
data_b: array of type integer containing 0-1 entries of shape [n=m*bits_per_symbol]
"""
function integers_to_bits(data_si, bits_per_symbol)
    bits = similar(data_si, length(data_si) * bits_per_symbol)
    for (offset, shift) in enumerate((bits_per_symbol - 1):-1:0)
        bits[offset:bits_per_symbol:end] .= (data_si .>> shift) .& 0x1
    end
    bits
end


"""
Converts array of integer representation of bits to bit symbol representation
Inputs:
data_si: array of type integer containing integer representation of rows of data_sb,
         of shape [m]
bits_per_symbol: scalar
Output:
data_sb: array of type integer containing 0-1 entries of shape [bits_per_symbol, m]
"""
function integers_to_symbols(data_si, bits_per_symbol)
    data_si = convert.(UInt16, data_si)
    symbs = similar(data_si, (bits_per_symbol, length(data_si)))
    for (row, shift) in enumerate((bits_per_symbol - 1):-1:0)
        symbs[row, :] .= (data_si .>> shift) .& 0x1
    end
    symbs
end


"""
Converts array of bits into array of integer representation of symbol
Inputs:
data_b: array of type integer containing 0-1 entries of shape [n]
bits_per_symbol: scalar such that n is divisible by bits_per_symbol
Output:
data_si: array of type integer containing integer representation of rows of data_sb,
         of shape [m=n/bits_per_symbol]
"""
function bits_to_integers(data_b, bits_per_symbol)
    data_sb = bits_to_symbols(data_b, bits_per_symbol)
    data_si = symbols_to_integers(data_sb)
    data_si
end


"""
Converts array of bits into array of bit representation of symbol
Inputs:
data_b: array of type integer containing 0-1 entries of shape [n]
bits_per_symbol: scalar such that n is divisible by bits_per_symbol
Output:
data_sb: array of type integer containing 0-1 entries of shape [bits_per_symbol, m=n/bits_per_symbol]
"""
function bits_to_symbols(data_b, bits_per_symbol)
    reshape(data_b, (bits_per_symbol, :))
end


"""
Converts array of bit represented symbols into integer represented symbols
Inputs:
data_sb: array of type integer containing 0-1 entries of shape [bits_per_symbol, m]
Output:
data_si: array of type integer containing integer representation of rows of data_sb,
         of shape [m]
"""
function symbols_to_integers(data_sb)
    nbits = size(data_sb, 1)
    powers = 2 .^ (nbits-1:-1:0)
    data_si = sum(data_sb .* powers, dims=1)[:]
    data_si
end


"""
Converts array of bit representation of symbols to array of bits
Inputs:
data_sb: array of type integer containing 0-1 entries of shape [bits_per_symbol, m]
Output:
data_b: array of type integer containing 0-1 entries of shape [n=m*bits_per_symbol]
"""
function symbols_to_bits(data_sb)
    data_sb[:]
end


"""
Converts complex numbers to 2D cartesian representation
Inputs:
data_c: array of type complex of shape [N]
Output:
data_d: array of type float of shape [2,N]
"""
function complex_to_cartesian(data_c)
    data_d = zeros(Float32, (2, length(data_c)))
    data_d[1, :] = real.(data_c)
    data_d[2, :] = imag.(data_c)
    data_d
end


"""
Converts 2D cartesian representation to complex numbers
Inputs:
data_c: array of type float of shape [2,N]
Output:
data_d: array of type complex of shape [N]
"""
function cartesian_to_complex(data_d)
    data_c = data_d[1, :] .+ 1im .* data_d[2, :]
    data_c
end


end