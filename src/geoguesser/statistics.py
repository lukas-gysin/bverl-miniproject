import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as stats

class Statistics:
    def __init__(self, dataset, batch_size=64, subset_ratio=None):
        self.dataset = dataset
        self.batch_size = batch_size

        # If subset_ratio is provided, subsample the dataset
        if subset_ratio:
            self.dataset = self._subset_dataset(subset_ratio)

    def _subset_dataset(self, subset_ratio):
        """Subsamples the dataset to reduce processing time."""
        subset_size = int(len(self.dataset) * subset_ratio)
        indices = torch.randperm(len(self.dataset))[:subset_size]
        return torch.utils.data.Subset(self.dataset, indices)

    def calculate_statistics(self, indices, subsample_ratio=1.0):
        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=os.cpu_count())

        channel_sum = None
        channel_sum_sq = None
        channel_min = None
        channel_max = None
        total_pixels = 0

        skewness = None
        kurtosis = None
        correlations = None

        # Adjust indices for subsampling
        indices = indices[:int(len(indices) * subsample_ratio)]

        for batch_idx, batch in enumerate(loader):
            if batch_idx not in indices:
                continue

            images = batch["image"].to(dtype=torch.float32)  # Ensure float32 for efficiency
            if channel_sum is None:
                # Initialize accumulators
                channel_sum = torch.zeros(images.size(1))
                channel_sum_sq = torch.zeros(images.size(1))
                channel_min = torch.full((images.size(1),), float("inf"))
                channel_max = torch.full((images.size(1),), float("-inf"))
                skewness = torch.zeros(images.size(1))
                kurtosis = torch.zeros(images.size(1))
                correlations = torch.zeros((images.size(1), images.size(1)))

            # Update sums and sums of squares
            channel_sum += images.reshape(images.size(1), -1).sum(dim=1)
            channel_sum_sq += (images ** 2).reshape(images.size(1), -1).sum(dim=1)

            # Update min and max
            batch_min = images.reshape(images.size(1), -1).min(dim=1)[0]
            batch_max = images.reshape(images.size(1), -1).max(dim=1)[0]
            channel_min = torch.minimum(channel_min, batch_min)
            channel_max = torch.maximum(channel_max, batch_max)

            # Incrementally calculate skewness and kurtosis
            for i in range(images.size(1)):  # Iterate over channels
                channel_data = images[:, i, :, :].reshape(-1).cpu().numpy()
                skewness[i] += stats.skew(channel_data)
                kurtosis[i] += stats.kurtosis(channel_data)

            # Calculate correlations
            flattened = images.reshape(images.size(1), -1).cpu()
            corr_matrix = torch.corrcoef(flattened)
            correlations += corr_matrix

            # Update total pixel count
            total_pixels += images.size(0) * images.size(2) * images.size(3)

        # Calculate final statistics
        means = channel_sum / total_pixels
        stds = torch.sqrt((channel_sum_sq / total_pixels) - (means ** 2))
        correlations /= len(loader)

        print("Flattened data shape:", flattened.shape)
        print("Correlation matrix:\n", corr_matrix)   
        band1 = flattened[0].numpy()
        band3 = flattened[2].numpy()
        manual_corr = stats.pearsonr(band1, band3)
        print(f"Manual correlation between Band 1 and Band 3: {manual_corr}")
        return means, stds, channel_min, channel_max, skewness / len(loader), kurtosis / len(loader), correlations

    def report(self, stats):
        means, stds, mins, maxs, skewness, kurtosis, correlations = stats
        print(f"Means: {means}")
        print(f"Standard Deviations: {stds}")
        print(f"Minimums: {mins}")
        print(f"Maximums: {maxs}")
        print(f"Skewness: {skewness}")
        print(f"Kurtosis: {kurtosis}")
        print("Correlations:")
        print(correlations)

    def visualize(self, stats):
        means, stds, mins, maxs, skewness, kurtosis, correlations = stats

        # Convert statistics to DataFrame for easy visualization
        data = pd.DataFrame({
            "Channel": [f"Band {i+1}" for i in range(len(means))],
            "Mean": means.numpy(),
            "Std Dev": stds.numpy(),
            "Min": mins.numpy(),
            "Max": maxs.numpy(),
            "Skewness": skewness.numpy(),
            "Kurtosis": kurtosis.numpy(),
        })

        # Create separate plots for each statistic
        plt.figure(figsize=(18, 12))

        # Plot Mean
        plt.subplot(3, 3, 1)
        sns.barplot(x="Channel", y="Mean", data=data)
        plt.title("Mean per Channel")
        plt.xticks(rotation=45)
        plt.ylabel("Mean")

        # Plot Standard Deviation
        plt.subplot(3, 3, 2)
        sns.barplot(x="Channel", y="Std Dev", data=data)
        plt.title("Standard Deviation per Channel")
        plt.xticks(rotation=45)
        plt.ylabel("Std Dev")

        # Plot Min
        plt.subplot(3, 3, 3)
        sns.barplot(x="Channel", y="Min", data=data)
        plt.title("Minimum Value per Channel")
        plt.xticks(rotation=45)
        plt.ylabel("Min")

        # Plot Max
        plt.subplot(3, 3, 4)
        sns.barplot(x="Channel", y="Max", data=data)
        plt.title("Maximum Value per Channel")
        plt.xticks(rotation=45)
        plt.ylabel("Max")

        # Plot Range (Max - Min)
        data["Range"] = data["Max"] - data["Min"]
        plt.subplot(3, 3, 5)
        sns.barplot(x="Channel", y="Range", data=data)
        plt.title("Range (Max - Min) per Channel")
        plt.xticks(rotation=45)
        plt.ylabel("Range")

        # Plot Skewness
        plt.subplot(3, 3, 6)
        sns.barplot(x="Channel", y="Skewness", data=data)
        plt.title("Skewness per Channel")
        plt.xticks(rotation=45)
        plt.ylabel("Skewness")

        # Plot Kurtosis
        plt.subplot(3, 3, 7)
        sns.barplot(x="Channel", y="Kurtosis", data=data)
        plt.title("Kurtosis per Channel")
        plt.xticks(rotation=45)
        plt.ylabel("Kurtosis")

        plt.tight_layout()
        plt.show()

    def visualize_correlation(self, correlations, channel_names):
        """Visualize correlation matrix as a heatmap."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlations.numpy(), annot=True, fmt=".2f", cmap="coolwarm", xticklabels=channel_names, yticklabels=channel_names)
        plt.title("Correlation Between Channels")
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.tight_layout()
        plt.show()
