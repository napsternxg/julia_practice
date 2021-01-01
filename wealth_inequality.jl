### A Pluto.jl notebook ###
# v0.12.16

using Markdown
using InteractiveUtils

# ╔═╡ fc12b4b2-4914-11eb-3394-91705c2af64e
using DifferentialEquations

# ╔═╡ 41e9bbf0-4915-11eb-31ca-cd0cca5857fd
using Plots; pyplot() #plotly() # Using the Plotly backend

# ╔═╡ 58400c3e-4917-11eb-0470-e59f5694a101
using Statistics

# ╔═╡ 6f6e08c2-4c57-11eb-1457-9d215dc4370b
function get_vals(sol)
	sol_matrix = hcat(sol.u...)
	mean_values = mean(sol_matrix, dims=(1))
	n_steps = length(sol.t)
	below_mean = mean(sol_matrix .< mean_values, dims=1)
	rolling_mean = cumsum(below_mean, dims=(2))[1,:] ./collect(range(1.0, n_steps, step=1.0))
	sol_rescaled = deepcopy(sol)
	sol_rescaled_matrix = sol_matrix ./ mean_values
	#sol_rescaled.u = [r[:] for r in eachrow(sol_rescaled_matrix)]
	mean_rescaled_values = mean(sol_rescaled_matrix, dims=(1))
	max_rescaled_values = maximum(sol_rescaled_matrix, dims=(1))
	min_rescaled_values = minimum(sol_rescaled_matrix, dims=(1))
	median_rescaled_values = median(sol_rescaled_matrix, dims=(1))
	q90_rescaled_values = hcat([quantile(r[:], 0.9) for r in eachcol(sol_rescaled_matrix)]...)
	q10_rescaled_values = hcat([quantile(r[:], 0.1) for r in eachcol(sol_rescaled_matrix)]...)
	rand_trajectory_indices = rand(1:n_steps, 5)
	vals = (
		mean_values, below_mean, rolling_mean, 
		mean_rescaled_values, max_rescaled_values, min_rescaled_values,
		median_rescaled_values, q90_rescaled_values, q10_rescaled_values,
		sol_rescaled_matrix, rand_trajectory_indices
	)
	return n_steps, vals
end

# ╔═╡ f47548f0-493a-11eb-2fa2-ef3ddcc41b8c
typeof(1/2^(4))

# ╔═╡ 42f50710-4939-11eb-36ac-c5106935d578
function get_diff_eq(;
		μ::Number=1.0, 
		σ::Number=1.0, 
		τ::Number=0.0, 
		n::Number=2, 
		tspan::Tuple{Number,Number}=(0.0, 100.0), 
		#u₀::Array{Number}=nothing
	)
	u₀ = ones(n)
	f(u,p,t) = μ*u - τ*(u .- mean(u))
	g(u,p,t) = σ*u
	prob = SDEProblem(f,g,u₀,tspan)
	return prob
end

# ╔═╡ 1db4c860-4915-11eb-0633-01cf38659378
begin
	μ=0.021
	σ=0.14
	τ=-0.15
	n=1000
	tspan=(0.0, 100.0)
	prob = get_diff_eq(μ=μ, σ=σ, τ=τ, n=n, tspan=tspan)
end

# ╔═╡ e45de3c0-4c5d-11eb-1d73-af09b6ea55e7
function generate_plot(sol, vals, indices; τ=τ)
	(
		mean_values, below_mean, rolling_mean, 
		mean_rescaled_values, max_rescaled_values, min_rescaled_values,
		median_rescaled_values, q90_rescaled_values, q10_rescaled_values,
		sol_rescaled_matrix, rand_trajectory_indices
	) = vals
	l = @layout [a; b; c]
	#indices = 1:i
	x = sol.t[indices]
	#p1 = plot(sol, yscale= :log, color=:gray, alpha=0.3, label="")
	p1 = plot(x, mean_values[1,indices], color=:red, label="mean")
	plot!(p1, legend=:outertopright, title="Mean wealth (τ=$τ, n=$n)")
	
	
	p3 = plot(x, below_mean[1,indices], label="below_mean")
	plot!(p3, x, rolling_mean[indices], label="time_avg")
	plot!(p3, legend=:outertopright, title="% below mean value")
	
	
	#p2 = plot(sol.t, sol_rescaled_matrix, color=:gray, alpha=0.3, label="")
	
	p2 = plot(x, mean_rescaled_values[1,indices], color=:red, label="mean")
	plot!(p2, x, max_rescaled_values[1,indices], color=:blue, label="max")
	plot!(p2, x, min_rescaled_values[1,indices], color=:blue, label="min")
	plot!(p2, x, median_rescaled_values[1,indices], color=:blue, label="50%")
	plot!(p2, x, q90_rescaled_values[1,indices], color=:magenta, label="90%")
	plot!(p2, x, q10_rescaled_values[1,indices], color=:magenta, label="10%")
	for i in rand_trajectory_indices
		plot!(p2, x, sol_rescaled_matrix[i,indices], color=:grey, label="")
	end
	
	plot!(p2, legend=:outertopright, title="Rescaled wealth")	
	plot(p1, p2, p3, size=(1000, 600), layout = l)
end

# ╔═╡ db453ac0-4c64-11eb-1345-4b2d4f09a895
prob2 = get_diff_eq(μ=μ, σ=σ, τ=-τ, n=n, tspan=tspan)

# ╔═╡ 34a39a10-4915-11eb-0f7a-77826dcf2fcb
begin
	dt=0.1
	sol = solve(prob,EM(),dt=dt)
	sol2 = solve(prob2,EM(),dt=dt)
end

# ╔═╡ 3a614340-4c66-11eb-1ce7-71a905665740
n_steps, vals  = get_vals(sol)

# ╔═╡ b5bfd142-4915-11eb-0d52-7319a94bf116
begin
	anim = @animate for i in range(10, n_steps, step=10)
		i = min(i, n_steps)
		generate_plot(sol, (vals), 1:i)
	end
	gif(anim, "anim_fps15.gif", fps = 10)
end

# ╔═╡ fc65033e-4c5d-11eb-3d9a-8f2c30930be4
generate_plot(sol, (vals), :)

# ╔═╡ 563fb410-4c67-11eb-1e98-e99fdd84d5c5
n_steps2, vals2 = get_vals(sol2)

# ╔═╡ 06148cae-4c65-11eb-0a03-d1a1806d27a7
begin
	anim2 = @animate for i in range(10, n_steps, step=10)
		i = min(i, n_steps)
		generate_plot(sol2, (vals2), 1:i, τ=-τ)
	end
	gif(anim2, "anim2_fps15.gif", fps = 10)
end

# ╔═╡ 9637f6b0-4c65-11eb-0671-fde2a1f4f4d9
generate_plot(sol2, (vals2), :, τ=-τ)

# ╔═╡ b6fa0250-493e-11eb-0069-5bc294ed5554
typeof(hcat(sol.u...))

# ╔═╡ 3b90c930-495e-11eb-2362-9fc243f491a0
rand(1:n_steps, 10)

# ╔═╡ 4eb210c0-4961-11eb-2e30-11f11dec964a
html"""<style>
main {
    margin: auto;
	max-width: 1400px;
}
"""

# ╔═╡ Cell order:
# ╠═b5bfd142-4915-11eb-0d52-7319a94bf116
# ╠═06148cae-4c65-11eb-0a03-d1a1806d27a7
# ╠═fc65033e-4c5d-11eb-3d9a-8f2c30930be4
# ╠═9637f6b0-4c65-11eb-0671-fde2a1f4f4d9
# ╠═e45de3c0-4c5d-11eb-1d73-af09b6ea55e7
# ╠═3a614340-4c66-11eb-1ce7-71a905665740
# ╠═563fb410-4c67-11eb-1e98-e99fdd84d5c5
# ╠═6f6e08c2-4c57-11eb-1457-9d215dc4370b
# ╠═1db4c860-4915-11eb-0633-01cf38659378
# ╠═db453ac0-4c64-11eb-1345-4b2d4f09a895
# ╠═fc12b4b2-4914-11eb-3394-91705c2af64e
# ╠═41e9bbf0-4915-11eb-31ca-cd0cca5857fd
# ╠═58400c3e-4917-11eb-0470-e59f5694a101
# ╠═f47548f0-493a-11eb-2fa2-ef3ddcc41b8c
# ╠═42f50710-4939-11eb-36ac-c5106935d578
# ╠═34a39a10-4915-11eb-0f7a-77826dcf2fcb
# ╠═b6fa0250-493e-11eb-0069-5bc294ed5554
# ╠═3b90c930-495e-11eb-2362-9fc243f491a0
# ╠═4eb210c0-4961-11eb-2e30-11f11dec964a
