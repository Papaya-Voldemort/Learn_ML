<script lang="ts">
	import { page } from '$app/state';
	import { resolve } from '$app/paths';

	const navItems = [
		{ name: 'Home', url: '/' },
		{ name: 'Demo', url: '/demo' },
		{ name: 'File to ASCII', url: '/file-converter' },
		{ name: 'Download', url: '/download' }
	] as const;
</script>

<nav class="navbar">
	<div class="nav-container">
		<a href={resolve('/')} class="logo">
			&lt;<span class="logo-accent">GhostChar</span> /&gt;
		</a>
		<div class="nav-links">
			{#each navItems as navItem (navItem.url)}
				<a
					href={resolve(navItem.url)}
					class={{ active: page.url.pathname === resolve(navItem.url) }}>{navItem.name}</a
				>
			{/each}
		</div>
	</div>
</nav>

<style>
	.navbar {
		display: flex;
		justify-content: center;
		width: 100%;
		background: var(--surface-container-lowest);
	}

	.nav-container {
		display: flex;
		align-items: center;
		justify-content: space-between;
		width: 100%;
		max-width: 1200px;
		padding: 1rem 2rem;
		font-family: var(--font-family);
	}

	.nav-links {
		display: flex;
		gap: 1.5rem;
	}

	.logo {
		font-family: 'Courier New', Courier, monospace;
		font-weight: 700;
		font-size: 1.2rem;
		color: var(--on-surface-variant);
		text-decoration: none;
		letter-spacing: -0.02em;
		transition: color 0.2s;
	}

	.logo:hover {
		color: var(--on-surface);
	}

	.logo-accent {
		color: var(--primary);
	}

	.nav-links a {
		position: relative;
		color: var(--on-surface-variant);
		text-transform: uppercase;
		text-decoration: none;
		letter-spacing: 0.05em;
		padding: 0.5rem 0;
		font-size: 0.9rem;
		font-weight: 500;
		transition:
			color 0.2s,
			border-color 0.2s;
	}

	.nav-links a::after {
		content: '';
		position: absolute;
		bottom: 0;
		left: 0;
		width: 100%;
		height: 2px;
		background-color: var(--primary);
		transform: scaleX(0);
		transform-origin: bottom left;
		transition: transform 0.25s ease-out;
	}

	.nav-links a:hover {
		color: var(--on-surface);
	}

	.nav-links a:hover::after {
		transform: scaleX(1);
	}

	.nav-links a.active {
		color: var(--primary);
	}

	.nav-links a.active::after {
		transform: scaleX(1);
		background-color: var(--primary);
	}
</style>
