import _ from 'lodash'


export async function queryPOI(args, state) {
	console.log(`[查询POI] 和风查询高德地图:  ${args.location}`);
	const res = await fetch(`https://restapi.amap.com/v3/place/text?key=${state.modelConfig.gaodeToken}&types=110000&keywords=${encodeURIComponent(args.location)}`, {
	})
	const results = await res.json()
	console.log(JSON.stringify(results, null, 2))
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

export async function queryWeather(args, state){
	const res = await fetch(`https://mk4wctabd2.re.qweatherapi.com/v7/weather/7d?location=${args.location}`, {
		headers: {
			'X-QW-Api-Key': state.modelConfig.qweatherToken,
		}
	})

	const results = await res.json()

	if(results.daily?.length > 0){
		return results.daily.map((day) => ({
			day: day.fxDate,
			tempMax: day.tempMax,
			tempMin: day.tempMin,
			textDay: day.textDay,
			windScaleDay: day.windScaleDay
		}))
	} else {
		return null
	}
}

export async function qWeather(args, state){
	console.log(`[工具调用] 和风天气: 查询 ${args.location} 未来7天的天气`);
	const location = await queryPOI(args, state);
	console.log(`坐标: ${location.location}, 名称: ${location.name}`);
	const weather = await queryWeather({ location: location.location }, state);
	if(weather){
		let weatherInfo = `未来7天${args.location}的天气情况如下：\n`
		weather.forEach((day)=>{
			weatherInfo += `${day.day}：最高气温 ${day.tempMax}°C，最低气温 ${day.tempMin}°C，天气 ${day.textDay}，风力 ${day.windScaleDay}级。\n`
		})
		return weatherInfo
	} else {
		return `无法获取${location.name}的天气信息，请稍后再试。`
	}
}
