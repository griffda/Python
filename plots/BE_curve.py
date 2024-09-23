import matplotlib.pyplot as plt
import numpy as np

# Sample data for binding energy per nucleon (in MeV) vs mass number (A)
# This is a simplified representation
mass_numbers = np.arange(1, 240)
binding_energies = 15.75 * mass_numbers - 17.8 * mass_numbers**(2/3) - 0.711 * mass_numbers**(1/3) + 23.7 * (mass_numbers % 2 == 0) / mass_numbers**0.5

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(mass_numbers, binding_energies, label='Binding Energy per Nucleon')

# Highlight regions for fusion and fission
fusion_region = mass_numbers[mass_numbers < 56]
fission_region = mass_numbers[mass_numbers > 56]

plt.fill_between(fusion_region, binding_energies[mass_numbers < 56], color='blue', alpha=0.3, label='Fusion Region')
plt.fill_between(fission_region, binding_energies[mass_numbers > 56], color='red', alpha=0.3, label='Fission Region')

# Add labels and title
plt.xlabel('Mass Number (A)')
plt.ylabel('Binding Energy per Nucleon (MeV)')
plt.title('Binding Energy Curve for Nuclear Fusion and Fission')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()