import _ from 'lodash'

export async function queryPOI(args, state) {
	console.log(`[查询POI] 高德地图: ${args.type} - ${args.location}`);
	const typeCode = args.type === 'scenic' ? '110000' : '120000';
	const res = await fetch(`https://restapi.amap.com/v3/place/text?key=${state.modelConfig.gaodeToken}&types=${typeCode}&keywords=${encodeURIComponent(args.location)}`, {
	})
	const results = await res.json()
	console.log(`[查询结果] 高德地图: ${JSON.stringify(results)}`);
	if(results?.pois?.length > 0){
		const searchLocation = _.find(results.pois, {name: args.location})
		if(searchLocation){
			return {
				name: searchLocation.name,
				id: searchLocation.id,
				city: searchLocation.cityname,
				address: searchLocation.address,
				location: searchLocation.location
			}
		} else {
			return {
				name: results.pois[0].name,
				id: results.pois[0].id,
				city: results.pois[0].cityname,
				address: results.pois[0].address,
				location: results.pois[0].location
			}
		}
	}
	return null
}

export async function queryDriveRoute(args, state) {
	const res = await fetch(`https://restapi.amap.com/v5/direction/driving?key=${state.modelConfig.gaodeToken}&origin=${encodeURIComponent(args.origin)}&destination=${encodeURIComponent(args.destination)}&strategy=0&show_fields=cost`, {
	})
	const results = await res.json()
	if(results?.route?.paths?.length > 0){
		return {
			distance: results.route.paths[0].distance / 1000, // 转换为公里
			duration: Math.floor(results.route.paths[0].cost.duration / 60), // 转换为分钟
			tolls: parseInt(results.route.paths[0].cost.tolls),
			traffic_lights: parseInt(results.route.paths[0].cost.traffic_lights)
		}
	} else {
		return null
	}
}

export async function gaodeMaps(args, state){
	console.log(`[工具调用] 高德地图: 从 ${args.start} 到 ${args.end}, 类型: ${args.type}`);
	const startArgs = { location: args.start}
	const endArgs = { location: args.end }
	if(args.type === 'forward'){
		startArgs.type = 'community'
		endArgs.type = 'scenic'
	} else {
		startArgs.type = 'scenic'
		endArgs.type = 'community'
	}
	const startLocation = await queryPOI(startArgs, state);
	const endLocation = await queryPOI(endArgs, state);
	if(startLocation && endLocation){
		const driveRoute = await queryDriveRoute({ origin: startLocation.location, destination: endLocation.location }, state);
		if(driveRoute){
			return `从 ${args.start} 到 ${args.end} 驾车总共需要经过 ${driveRoute.distance} 公里，预计耗时 ${driveRoute.duration} 分钟。过路费为 ${driveRoute.tolls} 元，预计会经过 ${driveRoute.traffic_lights} 个红绿灯。`;
		} else {
			return '无法计算起点和终点之间的驾车路线';
		}
	} else {
		return '无法找到起点或终点的位置信息';
	}
}
