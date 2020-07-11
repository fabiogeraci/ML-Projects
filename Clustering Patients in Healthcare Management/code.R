#### Libraries ####

library(factoextra)
library(NbClust)
library(ggplot2)
library(gridExtra)
library(ISLR)
library(cluster)
library(reshape2)
library(dplyr)
library(skimr)
library(mclust)
library(poLCA)
library(FactoMineR)


#### Importing Data ####


raw_data <- read.csv('C:/Users/Pritam/Desktop/R Projects/patient.csv')




#### Handling Missing values ####


# Copying 22 life variables into a new dataframe
data <- raw_data[, -c(1, 24, 25, 26)]

# Converting all -9 values into NA
for (col in 1:22) {
  data[data[, col] == -9, col] <- NA
}

# Droping all rows with NA values 
data <- na.omit(data)



#### K-Means Clustering ####
#K-Means with 4 clusters and 200 times with different starting values
set.seed(55)
kmeans_clusters <- kmeans(x = data, centers = 4, iter.max = 100, nstart = 500)

kmeans_clusters$centers

kmeans_clusters$tot.withinss

kmeans_clusters$size

kmeans_clusters$centers

#PCA Analysis
#Explained variance for different components in PCA
pca <- PCA(data,  graph = FALSE)
# Visualize eigenvalues/variances
fviz_screeplot(pca, addlabels = TRUE, ylim = c(0, 50))

var <- get_pca_var(pca)
# Contributions of variables to PC1 or first component
fviz_contrib(pca, choice = "var", axes = 1, top = 10)

#Top 8 feature contribution based on first two components
fviz_pca_var(pca, col.var="contrib", select.var = list(contrib = 8),
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE # Avoid text overlapping
) + theme_minimal() + ggtitle("Variables - PCA")




fviz_cluster(data = data, object = kmeans_clusters)
# Creating dataframe to transform the data into columns containing means for each cluster
kmeans_cluster_centers <- as.data.frame(kmeans_clusters$centers)

categories    <- as.data.frame(t(kmeans_cluster_centers))
categories$item <- rownames(categories)
# Visualisation for treating the means for each cluster as trajectories 
# Plotting the means of kmeans clusters
par(mar=c(7,4,3,3))
matplot(categories, type = "b", axes = F)
axis(2, tck = -.015, las = 1)
axis(side=1, at=1:nrow(categories),tck = -.015, labels=categories$item, las = 2)

a <- kmeans_clusters$centers
b <- t(a)

#### Partitioning Around Medoids(PAM) Clustering ####
# PAM with 4 clusters & 200 different starting values

set.seed(55)
pam_clusters <- clara(x = data, k = 4, metric = "manhattan", stand = FALSE,
                   samples = 200, pamLike = TRUE)

# PAM Means
mu <- pam_clusters$medoids

# Objective 
pam_clusters$clusinfo

# Visualisation of PAM clusters
fviz_cluster(pam_clusters, data=data)

# Creating dataframe to transform the data into columns containing means for each cluster
centers_pam <- as.data.frame(pam_clusters$medoids)
#Transformation
cat_pam      <- as.data.frame(t(centers_pam))
cat_pam$item <- rownames(cat_pam)

# Visualisation for treating the means for each cluster as trajectories 
# Plotting the means of PAM clusters
par(mar=c(7,4,3,3))
matplot(cat_pam, type = "b", axes = F)
axis(2, tck = -.015, las = 1)
axis(side=1, at=1:nrow(cat_pam),tck = -.015, labels=cat_pam$item, las = 2)



#### Optimal Number of Cluster Selection ####

# Elbow method
fviz_nbclust(data, kmeans, method = "wss") 
  #geom_vline(xintercept = 4, linetype = 2)+
  #labs(subtitle = "Elbow method")

# Gap statistic
set.seed(123)
fviz_nbclust(data, kmeans, nstart = 25,  method = "gap_stat", nboot = 500)
  #labs(subtitle = "Gap statistic method")

sil_km <- fviz_nbclust(data, FUN = kmeans, method = "silhouette")
plot(sil_km)



#Average silhoutte width to determine optimal number of Clusters
# Packaging function for PAM
pam_packaging_function <- function(x,k)
  list(cluster = pam(x, k, metric="manhattan", cluster.only=TRUE))


silhoutte <- fviz_nbclust(data, FUN = pam_packaging_function, method = "silhouette")

silhoutte


#Dendrogram
clust_hier <- hcut(data, k = 7, hc_method = "complete")
# Visualize dendrogram
fviz_dend(clust_hier, show_labels = TRUE, rect = TRUE)

#Nbclust indices
clust_indices <- NbClust(data, distance = "euclidean",
        min.nc = 2, max.nc = 9, 
        method = "complete", index ="all")
factoextra::fviz_nbclust(res.nbclust) + theme_minimal() + ggtitle("NbClust's optimal number of clusters")

#### PAM & K-means with OPTIMAL number of clusters  #### 



# PAM with OPTIMAL number of clusters
set.seed(55)
pam_clusters_refined <- clara(x = data, k = 2, metric = "manhattan", stand = FALSE,
                   samples = 200, pamLike = TRUE)

fviz_cluster(pam_clusters_refined, data=data)

# Creating dataframe to transform the data into columns containing means for each cluster
centers_pam_new <- as.data.frame(pam_clusters_refined$medoids)

rs_pam_new     <- as.data.frame(t(centers_pam_new))
rs_pam_new$item <- rownames(rs_pam_new)


# Visualisation for treating the means for each cluster as trajectories 
# Plotting the means of PAM clusters
par(mar=c(7,4,3,3))
matplot(rs_pam_new, type = "b", axes = F)
axis(2, tck = -.015, las = 1)
axis(side=1, at=1:nrow(rs_pam_new),tck = -.015, labels=rs_pam_new$item, las = 2)

# Kmeans with OPTIMAL number of clusters
set.seed(55)
kmeans_clusters_refined <- kmeans(x = data, centers = 7, iter.max = 100, nstart = 200)

kmeans_clusters_refined$size

kmeans_clusters_refined

fviz_cluster(data = data, object = kmeans_clusters_refined)

# Creating dataframe to transform the data into columns containing means for each cluster
kmeans_cluster_centers_new <- as.data.frame(kmeans_clusters_refined$centers)

rs_new      <- as.data.frame(t(kmeans_cluster_centers_new))
rs_new$item <- rownames(rs_new)

# Visualisation for treating the means for each cluster as trajectories 
# Plotting the means of KMEANS clusters
par(mar=c(7,4,3,3))
matplot(rs_new, type = "b", axes = F)
axis(2, tck = -.015, las = 1)
axis(side=1, at=1:nrow(rs_new),tck = -.015, labels=rs_new$item, las = 2)


#### Model Based Clustering -  Gaussian mixture model (GMM) ####


set.seed(55)

gmm_clust <- mclustBIC(data)

gmm_clust



fit_gmm <- Mclust(data, x = gmm_clust)

summary(fit_gmm, parameters = T)  

# plotting classification of data to clusters
plot(fit_gmm, what = "classification")

# Plotting the estimated pdf run in GMM
plot(fit_gmm, what = "density",  main = "")

fit_gmm$classification

fit_gmm$parameters$variance$sigma

# Visualisation of clusters in GMM
fviz_mclust(fit_gmm, "classification", geom = "point", 
            pointsize = 1.5, palette = "jco")


#### Latent Class Analysis ####

#Data copied to a new dataframe from raw_data as all variables will be treated as categorical for LCA
data_lca <- raw_data[, -c(1)]

#Handling missing values in this case
for (col in 1:25) {
  data_lca[data_lca[, col] == -9, col] <- NA
}

data_lca <- na.omit(data_lca)

# Converting data types into factors except features~ age and relationship
for (col in 1:23) {
  data_lca[, col] <- as.factor(data_lca[, col])
}

# Fitting appropriate model
K = 5
# poLCA Analysis
metrics <- matrix(,K,4)
for (k in 1:K){
  poLCA_fit <-
    poLCA(
      cbind(
        Work,
        Hobby,
        Breath,
        Pain,
        Rest,
        Sleep,
        Appetite,
        Nausea,
        Vomit,
        Constipated,
        Diarrhoea,
        Tired,
        Interfere,
        Concentrate,
        Tense,
        Worry,
        Irritate,
        Depressed,
        Memory,
        Family,
        Social,
        Financial
      ) ~ 1,
      maxiter = 50000,
      nclass = k,
      nrep = 50,
      data = data_lca
    )
  metrics[k,] <- c(k, poLCA_fit$llik,poLCA_fit$bic,poLCA_fit$aic)
}
# Format and display metrics
metrics <- as.data.frame(metrics)
colnames(metrics) <- cbind('K','Log-Likelihood','BIC','AIC')
print(metrics)

# K=3 minimises BIC value
# Fitting with 3 classes

poLCA_fit_final <-
  poLCA(
    cbind(
      Work,
      Hobby,
      Breath,
      Pain,
      Rest,
      Sleep,
      Appetite,
      Nausea,
      Vomit,
      Constipated,
      Diarrhoea,
      Tired,
      Interfere,
      Concentrate,
      Tense,
      Worry,
      Irritate,
      Depressed,
      Memory,
      Family,
      Social,
      Financial
    ) ~ 1,
    maxiter = 50000,
    nclass = 3,
    nrep = 20,
    data = data_lca
  )


# Visualisation
plot(poLCA_fit_final)

# Including covariates

poLCA_fit_final_cov <-
  poLCA(
    cbind(
      Work,
      Hobby,
      Breath,
      Pain,
      Rest,
      Sleep,
      Appetite,
      Nausea,
      Vomit,
      Constipated,
      Diarrhoea,
      Tired,
      Interfere,
      Concentrate,
      Tense,
      Worry,
      Irritate,
      Depressed,
      Memory,
      Family,
      Social,
      Financial
    ) ~ Sex,
    maxiter = 50000,
    nclass = 3,
    nrep = 20,
    data = data_lca
  )

poLCA_fit_final_cov