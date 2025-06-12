

export async function queryPOI(queryString, config) {
	const res = await fetch(`mk4wctabd2.re.qweatherapi.com`, {
		headers: {
			'Authorization': `Bearer ${config.QWEATHER_API_KEY}`,
		}
	})
	return res.json()?.poi?.map((poi)=>({
		name: poi.name,
		id: poi.id,
		city: poi.adm1
	}))
}
