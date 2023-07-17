---
title: Surfstore - A dropbox-like service
summary: Built a scalable, distributed Dropbox-like cloud service for file syncing, with fault-tolerance. 

tags:
- Systems
date: 2023-02-01
---


A fun course project I did as a part of the Graduate Networked Systems course at UCSD. We created a cloud-based file storage service called SurfStore in Golang. SurfStore is a networked file storage application based on Dropbox, that lets you sync files to and from the “cloud”. We implemented the cloud service, and a client which interacts with the service via gRPC. SurfStore has two services: the BlockStore (which stores file blocks, each of which is accessible via an API call by a unique identifier), and the MetaStore (which stores metadata of the files and the entire system). Each file gets broken into blocks, and these blocks are stored in the BlockStore, with the file metadata (file version, list of blocks ids, etc) in the MetaStore. Versioning helps in resolving conflicts, and in making sure the client first "syncs" their local filesystem before uploading to the cloud. We also implemented horizontal scaling ( partitioning file blocks among multiple Blockstores ) using [consistent hashing](https://en.wikipedia.org/wiki/Consistent_hashing). Consistent hashing is basically one way of making sure we have a straightforward way of sending a block to a specific Blockstore, and there's very little data transfer required when you add/remove a Blockstore. We further added fault-tolerance for the MetaStore service (making sure that there is a consistent update in case of node failures) by implementing the [RAFT protocol](https://raft.github.io/). 

This was my first time programming in Golang, and very enjoyable! 
